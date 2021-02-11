"""Adapted version of the NeRF example from pytorch3d. Includes a Bayesian version of NeRF based on TyXe.
The original notebook is available at: https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/fit_simple_neural_radiance_field.ipynb"""

# LICENSE FROM THE pytorch3D repo:
# --------------------------------
# BSD 3-Clause License
#
# For PyTorch3D software
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name Facebook nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# END OF LICENSE


import argparse
from functools import partial
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import namedtuple

# Data structures and functions for rendering
try:
    from pytorch3d.io import load_objs_as_meshes
    from pytorch3d.structures import Volumes
    from pytorch3d.transforms import so3_exponential_map
    from pytorch3d.renderer import (
        BlendParams,
        EmissionAbsorptionRaymarcher,
        FoVPerspectiveCameras,
        ImplicitRenderer,
        look_at_view_transform,
        MeshRasterizer,
        MeshRenderer,
        MonteCarloRaysampler,
        NDCGridRaysampler,
        PointLights,
        RasterizationSettings,
        RayBundle,
        ray_bundle_to_ray_points,
        SoftPhongShader,
        SoftSilhouetteShader,
    )
except ImportError:
    print("Failed to import from pytorch3d. This is not a dependency of Tyxe and needs to be installed separately. "
          "You may need to install the nightly rather than the stable version")
    raise

import pyro
import pyro.distributions as dist


import tyxe


# create the default data directory
current_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(current_dir, "..", "data", "cow_mesh")


def generate_cow_renders(
    num_views: int = 40, data_dir: str = DATA_DIR, azimuth_low: float = -180, azimuth_high: float = 180
):
    """
    This function generates `num_views` renders of a cow mesh.
    The renders are generated from viewpoints sampled at uniformly distributed
    azimuth intervals. The elevation is kept constant so that the camera's
    vertical position coincides with the equator.
    For a more detailed explanation of this code, please refer to the
    docs/tutorials/fit_textured_mesh.ipynb notebook.
    Args:
        num_views: The number of generated renders.
        data_dir: The folder that contains the cow mesh files. If the cow mesh
            files do not exist in the folder, this function will automatically
            download them.
    Returns:
        cameras: A batch of `num_views` `FoVPerspectiveCameras` from which the
            images are rendered.
        images: A tensor of shape `(num_views, height, width, 3)` containing
            the rendered images.
        silhouettes: A tensor of shape `(num_views, height, width)` containing
            the rendered silhouettes.
    """

    # set the paths

    # download the cow mesh if not done before
    cow_mesh_files = [
        os.path.join(data_dir, fl) for fl in ("cow.obj", "cow.mtl", "cow_texture.png")
    ]
    if any(not os.path.isfile(f) for f in cow_mesh_files):
        os.makedirs(data_dir, exist_ok=True)
        os.system(
            f"wget -P {data_dir} "
            + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj"
        )
        os.system(
            f"wget -P {data_dir} "
            + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl"
        )
        os.system(
            f"wget -P {data_dir} "
            + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png"
        )

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Load obj file
    obj_filename = os.path.join(data_dir, "cow.obj")
    mesh = load_objs_as_meshes([obj_filename], device=device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
    # to its original center and scale.  Note that normalizing the target mesh,
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-(center.expand(N, 3)))
    mesh.scale_verts_((1.0 / float(scale)))

    # Get a batch of viewing angles.
    elev = torch.linspace(0, 0, num_views)  # keep constant
    azim = torch.linspace(azimuth_low, azimuth_high, num_views) + 180.0

    # Place a point light in front of the object. As mentioned above, the front of
    # the cow is facing the -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Initialize an OpenGL perspective camera that represents a batch of different
    # viewing angles. All the cameras helper methods support mixed type inputs and
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and
    # then specify elevation and azimuth angles for each viewpoint as tensors.
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output
    # image to be of size 128X128. As we are rendering images for visualization
    # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
    # rasterize_meshes.py for explanations of these parameters.  We also leave
    # bin_size and max_faces_per_bin to their default values of None, which sets
    # their values using huristics and ensures that the faster coarse-to-fine
    # rasterization method is used.  Refer to docs/notes/renderer.md for an
    # explanation of the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=128, blur_radius=0.0, faces_per_pixel=1
    )

    # Create a phong renderer by composing a rasterizer and a shader. The textured
    # phong shader will interpolate the texture uv coordinates for each vertex,
    # sample from a texture image and apply the Phong lighting model
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    # Create a batch of meshes by repeating the cow mesh and associated textures.
    # Meshes has a useful `extend` method which allows us do this very easily.
    # This also extends the textures.
    meshes = mesh.extend(num_views)

    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras, lights=lights)

    # Rasterization settings for silhouette rendering
    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=128, blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma, faces_per_pixel=50
    )

    # Silhouette renderer
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader(),
    )

    # Render silhouette images.  The 3rd channel of the rendering output is
    # the alpha/silhouette channel
    silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)

    # binary silhouettes
    silhouette_binary = (silhouette_images[..., 3] > 1e-4).float()

    return cameras, target_images[..., :3], silhouette_binary


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
    **kwargs
):
    """
    A util function for plotting a grid of images.
    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.
    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3], **kwargs)
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3], **kwargs)
        if not show_axes:
            ax.set_axis_off()


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**self.n_harmonic_functions * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**self.n_harmonic_functions * x[..., i])
            ]

        Note that `x` is also premultiplied by `omega0` before
        evaluting the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


def save_io_nerf(rays_points_world, rays_densities, rays_colors, path='./data/'):
    os.makedirs(path, exist_ok=True)
    torch.save(rays_points_world.detach().to("cpu"), os.path.join(path, 'rays_points_world.pt'))
    torch.save(rays_densities.detach().to("cpu"), os.path.join(path, "rays_densities.pt"))
    torch.save(rays_colors.detach().to("cpu"), os.path.join(path, "rays_colors.pt"))


class NeuralRadianceField(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, n_hidden_neurons=256):
        super().__init__()
        """
        Args:
            n_harmonic_functions: The number of harmonic functions
                used to form the harmonic embedding of each point.
            n_hidden_neurons: The number of hidden units in the
                fully connected layers of the MLPs of the model.
        """

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)

        # The dimension of the harmonic embedding.
        embedding_dim = n_harmonic_functions * 2 * 3

        # self.mlp is a simple 2-layer multi-layer perceptron
        # which converts the input per-point harmonic embeddings
        # to a latent representation.
        # Not that we use Softplus activations instead of ReLU.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        )

        # Given features predicted by self.mlp, self.color_layer
        # is responsible for predicting a 3-D per-point vector
        # that represents the RGB color of the point.
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
            # To ensure that the colors correctly range between [0-1],
            # the layer is terminated with a sigmoid layer.
        )

        # The density layer converts the features of self.mlp
        # to a 1D density value representing the raw opacity
        # of each point.
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
            # Sofplus activation ensures that the raw opacity
            # is a non-negative number.
        )

        # We set the bias of the density layer to - 1.5 in order to initialize the opacities of the
        # ray points to values close to 0.
        # This is a crucial detail for ensuring convergence of the model.
        self.density_layer[0].bias.data[0] = -1.5

    def _get_densities(self, features):
        """
        This function takes `features` predicted by `self.mlp`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later mapped to [0-1] range with
        1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()

    def _get_colors(self, features, rays_directions):
        """
        This function takes per-point `features` predicted by `self.mlp`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.

        In order to represent viewpoint dependent effects,
        before evaluating `self.color_layer`, `NeuralRadianceField`
        concatenates to the `features` a harmonic embedding
        of `ray_directions`, which are per-point directions
        of point rays expressed as 3D l2-normalized vectors
        in world coordinates.
        """
        spatial_size = features.shape[:-1]

        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding(
            rays_directions_normed
        )

        # Expand the ray directions tensor so that its spatial size
        # is equal to the size of features.
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )

        # Concatenate ray direction embeddings with
        # features and evaluate the color model.
        color_layer_input = torch.cat(
            (features, rays_embedding_expand),
            dim=-1)
        return self.color_layer(color_layer_input)

    def forward(
        self,
        ray_bundle: RayBundle,
        path = None,
        **kwargs,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacitiy of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding(
            rays_points_world
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]

        # Finally, given the per-point features,
        # execute the density and color branches.

        rays_densities = self._get_densities(features)
        # rays_densities.shape = [minibatch x ... x 1]

        rays_colors = self._get_colors(features, ray_bundle.directions)
        # rays_colors.shape = [minibatch x ... x 3]

        if path is not None:
            save_io_nerf(rays_points_world, rays_densities, rays_colors, path)

        return rays_densities, rays_colors


# TYXE comment: this was previously a method of the NRF class, I'm now passing a PytorchBNN instance as the net argument
def batched_forward(
    net,
    ray_bundle: RayBundle,
    n_batches: int = 16,
    path = None,
    **kwargs,
):
    """
    This function is used to allow for memory efficient processing
    of input rays. The input rays are first split to `n_batches`
    chunks and passed through the `self.forward` function one at a time
    in a for loop. Combined with disabling Pytorch gradient caching
    (`torch.no_grad()`), this allows for rendering large batches
    of rays that do not all fit into GPU memory in a single forward pass.
    In our case, batched_forward is used to export a fully-sized render
    of the radiance field for visualisation purposes.

    Args:
        ray_bundle: A RayBundle object containing the following variables:
            origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                origins of the sampling rays in world coords.
            directions: A tensor of shape `(minibatch, ..., 3)`
                containing the direction vectors of sampling rays in world coords.
            lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                containing the lengths at which the rays are sampled.
        n_batches: Specifies the number of batches the input rays are split into.
            The larger the number of batches, the smaller the memory footprint
            and the lower the processing speed.

    Returns:
        rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
            denoting the opacitiy of each ray point.
        rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
            denoting the color of each ray point.

    """

    # Parse out shapes needed for tensor reshaping in this function.
    n_pts_per_ray = ray_bundle.lengths.shape[-1]
    spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

    # Split the rays to `n_batches` batches.
    tot_samples = ray_bundle.origins.shape[:-1].numel()
    batches = torch.chunk(torch.arange(tot_samples), n_batches)

    # For each batch, execute the standard forward pass.
    batch_outputs = [
        net(
            RayBundle(
                origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                xys=None,
            ), path=None if path is None else f"{path}/batch{i}"
        ) for i, batch_idx in enumerate(batches)
    ]

    # Concatenate the per-batch rays_densities and rays_colors
    # and reshape according to the sizes of the inputs.
    rays_densities, rays_colors = [
        torch.cat(
            [batch_output[output_i] for batch_output in batch_outputs], dim=0
        ).view(*spatial_size, -1) for output_i in (0, 1)
    ]
    if path is not None:
        torch.save(spatial_size, f"{path}/spatial_size.pt")
    return rays_densities, rays_colors


def huber(x, y, scaling=0.1):
    """
    A helper function for evaluating the smooth L1 (huber) loss
    between the rendered silhouettes and colors.
    """
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling ** 2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss


def sample_images_at_mc_locs(target_images, sampled_rays_xy):
    """
    Given a set of Monte Carlo pixel locations `sampled_rays_xy`,
    this method samples the tensor `target_images` at the
    respective 2D locations.

    This function is used in order to extract the colors from
    ground truth images that correspond to the colors
    rendered using `MonteCarloRaysampler`.
    """
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]
    # In order to sample target_images, we utilize
    # the grid_sample function which implements a
    # bilinear image sampler.
    # Note that we have to invert the sign of the
    # sampled ray positions to convert the NDC xy locations
    # of the MonteCarloRaysampler to the coordinate
    # convention of grid_sample.
    images_sampled = torch.nn.functional.grid_sample(
        target_images.permute(0, 3, 1, 2),
        -sampled_rays_xy.view(ba, -1, 1, 2),  # note the sign inversion
        align_corners=True
    )
    return images_sampled.permute(0, 2, 3, 1).view(
        ba, *spatial_size, dim
    )


def show_full_render(
        neural_radiance_field, camera,
        target_image, target_silhouette,
        loss_history_color, loss_history_sil,
        renderer_grid, num_forward=1
):
    """
    This is a helper function for visualizing the
    intermediate results of the learning.

    Since the `NeuralRadianceField` suffers from
    a large memory footprint, which does not allow to
    render the full image grid in a single forward pass,
    we utilize the `NeuralRadianceField.batched_forward`
    function in combination with disabling the gradient caching.
    This chunks the set of emitted rays to batches and
    evaluates the implicit function on one-batch at a time
    to prevent GPU memory overflow.
    """

    rendered_image_list, rendered_silhouette_list = [], []
    # Prevent gradient caching.
    with torch.no_grad():
        for _ in range(num_forward):
            # Render using the grid renderer and the
            # batched_forward function of neural_radiance_field.
            rendered_image_silhouette, _ = renderer_grid(
                cameras=camera,
                volumetric_function=partial(batched_forward, net=neural_radiance_field)
            )
            # Split the rendering result to a silhouette render
            # and the image render.
            rendered_image_, rendered_silhouette_ = (
                rendered_image_silhouette[0].split([3, 1], dim=-1)
            )
            rendered_image_list.append(rendered_image_)
            rendered_silhouette_list.append(rendered_silhouette_)

        rendered_images = torch.stack(rendered_image_list)
        rendered_image = rendered_images.mean(0)

        rendered_silhouettes = torch.stack(rendered_silhouette_list)
        rendered_silhouette = rendered_silhouettes.mean(0)

        if num_forward > 1:
            rendered_image_std = rendered_images.var(0).sum(-1).sqrt()
            rendered_silhouette_std = rendered_silhouettes.std(0)
        else:
            rendered_image_std = torch.zeros_like(rendered_image[..., 0])
            rendered_silhouette_std = torch.zeros_like(rendered_silhouette)

    print(f"Max image std: {rendered_image_std.max().item():.4f}; "
          f"max image: {rendered_image.max().item():.4f}; "
          f"max silhouette std: {rendered_silhouette_std.max().item():.4f}; "
          f"max silhouette: {rendered_silhouette.max().item():.4f}")
    # Generate plots.
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    ax = ax.ravel()
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    ax[0].plot(list(range(len(loss_history_color))), loss_history_color, linewidth=1)
    ax[1].imshow(clamp_and_detach(rendered_image))
    ax[2].imshow(clamp_and_detach(rendered_silhouette[..., 0]))
    ax[3].imshow(clamp_and_detach(rendered_image_std), cmap="hot", vmax=0.75 ** 0.5)
    ax[4].plot(list(range(len(loss_history_sil))), loss_history_sil, linewidth=1)
    ax[5].imshow(clamp_and_detach(target_image))
    ax[6].imshow(clamp_and_detach(target_silhouette))
    ax[7].imshow(clamp_and_detach(rendered_silhouette_std), cmap="hot", vmax=0.5)
    for ax_, title_ in zip(
            ax,
            (
                    "loss color", "rendered image", "rendered silhouette", "image uncertainty",
                    "loss silhouette", "target image", "target silhouette", "silhouette uncertainty"
            )
    ):
        if not title_.startswith('loss'):
            ax_.grid("off")
            ax_.axis("off")
        ax_.set_title(title_)
    fig.canvas.draw()
    fig.show()
    return fig


def generate_rotating_nerf(neural_radiance_field, target_cameras, renderer_grid, device, n_frames=50, num_forward=1, save_visualization=False):
    logRs = torch.zeros(n_frames, 3, device=device)
    logRs[:, 1] = torch.linspace(-3.14, 3.14, n_frames, device=device)
    Rs = so3_exponential_map(logRs)
    Ts = torch.zeros(n_frames, 3, device=device)
    Ts[:, 2] = 2.7
    frames = []
    uncertainties = []
    path = f"nerf_vis/view{i}/sample{j}" if save_visualization else None
    print('Rendering rotating NeRF ...')
    for i, (R, T) in enumerate(zip(tqdm(Rs), Ts)):
        camera = FoVPerspectiveCameras(
            R=R[None],
            T=T[None],
            znear=target_cameras.znear[0],
            zfar=target_cameras.zfar[0],
            aspect_ratio=target_cameras.aspect_ratio[0],
            fov=target_cameras.fov[0],
            device=device,
        )
        # Note that we again render with `NDCGridSampler`
        # and the batched_forward function of neural_radiance_field.
        frame_samples = torch.stack([renderer_grid(
                cameras=camera,
                volumetric_function=partial(batched_forward, net=neural_radiance_field, path=path)
            )[0][..., :3] for j in range(num_forward)])
        frames.append(frame_samples.mean(0))
        uncertainties.append(frame_samples.var(0).sum(-1).sqrt() if num_forward > 1 else torch.zeros_like(frame_samples[0, ..., 0]))
    return torch.cat(frames), torch.cat(uncertainties)


def main(inference, n_iter, save_state_dict, load_state_dict, kl_annealing_iters, zero_kl_iters, max_kl_factor,
         init_scale, save_visualization):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        print(
            'Please note that NeRF is a resource-demanding method.'
            + ' Running this notebook on CPU will be extremely slow.'
            + ' We recommend running the example on a GPU'
            + ' with at least 10 GB of memory.'
        )
        device = torch.device("cpu")

    target_cameras, target_images, target_silhouettes = generate_cow_renders(
        num_views=30, azimuth_low=-180, azimuth_high=90)
    print(f'Generated {len(target_images)} images/silhouettes/cameras.')

    # render_size describes the size of both sides of the
    # rendered images in pixels. Since an advantage of
    # Neural Radiance Fields are high quality renders
    # with a significant amount of details, we render
    # the implicit function at double the size of
    # target images.
    render_size = target_images.shape[1] * 2

    # Our rendered scene is centered around (0,0,0)
    # and is enclosed inside a bounding box
    # whose side is roughly equal to 3.0 (world units).
    volume_extent_world = 3.0

    # 1) Instantiate the raysamplers.

    # Here, NDCGridRaysampler generates a rectangular image
    # grid of rays whose coordinates follow the PyTorch3d
    # coordinate conventions.
    raysampler_grid = NDCGridRaysampler(
        image_height=render_size,
        image_width=render_size,
        n_pts_per_ray=128,
        min_depth=0.1,
        max_depth=volume_extent_world,
    )

    # MonteCarloRaysampler generates a random subset
    # of `n_rays_per_image` rays emitted from the image plane.
    raysampler_mc = MonteCarloRaysampler(
        min_x=-1.0,
        max_x=1.0,
        min_y=-1.0,
        max_y=1.0,
        n_rays_per_image=750,
        n_pts_per_ray=128,
        min_depth=0.1,
        max_depth=volume_extent_world,
    )

    # 2) Instantiate the raymarcher.
    # Here, we use the standard EmissionAbsorptionRaymarcher
    # which marches along each ray in order to render
    # the ray into a single 3D color vector
    # and an opacity scalar.
    raymarcher = EmissionAbsorptionRaymarcher()

    # Finally, instantiate the implicit renders
    # for both raysamplers.
    renderer_grid = ImplicitRenderer(
        raysampler=raysampler_grid, raymarcher=raymarcher,
    )
    renderer_mc = ImplicitRenderer(
        raysampler=raysampler_mc, raymarcher=raymarcher,
    )

    # First move all relevant variables to the correct device.
    renderer_grid = renderer_grid.to(device)
    renderer_mc = renderer_mc.to(device)
    target_cameras = target_cameras.to(device)
    target_images = target_images.to(device)
    target_silhouettes = target_silhouettes.to(device)

    # Set the seed for reproducibility
    torch.manual_seed(1)

    # Instantiate the radiance field model.
    neural_radiance_field_net = NeuralRadianceField().to(device)
    if load_state_dict is not None:
        sd = torch.load(load_state_dict)
        sd["harmonic_embedding.frequencies"] = neural_radiance_field_net.harmonic_embedding.frequencies
        neural_radiance_field_net.load_state_dict(sd)

    # TYXE comment: set up the BNN depending on the desired inference
    standard_normal = dist.Normal(torch.tensor(0.).to(device), torch.tensor(1.).to(device))
    prior_kwargs = {}
    test_samples = 1
    if inference == "ml":
        prior_kwargs.update(expose_all=False, hide_all=True)
        guide = None
    elif inference == "map":
        guide = partial(pyro.infer.autoguide.AutoDelta,
                        init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(neural_radiance_field_net))
    elif inference == "mean-field":
        guide = partial(tyxe.guides.AutoNormal, init_scale=init_scale,
                        init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(neural_radiance_field_net))
        test_samples = 8
    else:
        raise RuntimeError(f"Unreachable inference: {inference}")

    prior = tyxe.priors.IIDPrior(standard_normal, **prior_kwargs)
    neural_radiance_field = tyxe.PytorchBNN(neural_radiance_field_net, prior, guide)

    # TYXE comment: we need a batch of dummy data for the BNN to trace the parameters
    dummy_data = namedtuple("RayBundle", "origins directions lengths")(
        torch.randn(1, 1, 3).to(device),
        torch.randn(1, 1, 3).to(device),
        torch.randn(1, 1, 8).to(device)
    )
    # Instantiate the Adam optimizer. We set its master learning rate to 1e-3.
    lr = 1e-3
    optimizer = torch.optim.Adam(neural_radiance_field.pytorch_parameters(dummy_data), lr=lr)

    # We sample 6 random cameras in a minibatch. Each camera
    # emits raysampler_mc.n_pts_per_image rays.
    batch_size = 6

    # Init the loss history buffers.
    loss_history_color, loss_history_sil = [], []

    if kl_annealing_iters > 0 or zero_kl_iters > 0:
        kl_factor = 0.
        kl_annealing_rate = max_kl_factor / max(kl_annealing_iters, 1)
    else:
        kl_factor = max_kl_factor
        kl_annealing_rate = 0.
    # The main optimization loop.
    for iteration in range(n_iter):
        # In case we reached the last 75% of iterations,
        # decrease the learning rate of the optimizer 10-fold.
        if iteration == round(n_iter * 0.75):
            print('Decreasing LR 10-fold ...')
            optimizer = torch.optim.Adam(
                neural_radiance_field.pytorch_parameters(dummy_data), lr=lr * 0.1
            )

        # Zero the optimizer gradient.
        optimizer.zero_grad()

        # Sample random batch indices.
        batch_idx = torch.randperm(len(target_cameras))[:batch_size]

        # Sample the minibatch of cameras.
        batch_cameras = FoVPerspectiveCameras(
            R=target_cameras.R[batch_idx],
            T=target_cameras.T[batch_idx],
            znear=target_cameras.znear[batch_idx],
            zfar=target_cameras.zfar[batch_idx],
            aspect_ratio=target_cameras.aspect_ratio[batch_idx],
            fov=target_cameras.fov[batch_idx],
            device=device,
        )

        rendered_images_silhouettes, sampled_rays = renderer_mc(
            cameras=batch_cameras,
            volumetric_function=partial(batched_forward, net=neural_radiance_field)
        )
        rendered_images, rendered_silhouettes = (
            rendered_images_silhouettes.split([3, 1], dim=-1)
        )

        # Compute the silhoutte error as the mean huber
        # loss between the predicted masks and the
        # sampled target silhouettes.
        silhouettes_at_rays = sample_images_at_mc_locs(
            target_silhouettes[batch_idx, ..., None],
            sampled_rays.xys
        )
        sil_err = huber(
            rendered_silhouettes,
            silhouettes_at_rays,
        ).abs().mean()

        # Compute the color error as the mean huber
        # loss between the rendered colors and the
        # sampled target images.
        colors_at_rays = sample_images_at_mc_locs(
            target_images[batch_idx],
            sampled_rays.xys
        )
        color_err = huber(
            rendered_images,
            colors_at_rays,
        ).abs().mean()

        # The optimization loss is a simple
        # sum of the color and silhouette errors.
        # TYXE comment: we also add a kl loss for the variational posterior scaled by the size of the data
        # i.e. the total number of data points times the number of values that the data-dependent part of the
        # objective averages over. Effectively I'm treating this as if this was something like a Bernoulli likelihood
        # in a VAE where the expected log likelihood is averaged over both data points and pixels
        beta = kl_factor / (target_images.numel() + target_silhouettes.numel())
        kl_err = neural_radiance_field.cached_kl_loss
        loss = color_err + sil_err + beta * kl_err

        # Log the loss history.
        loss_history_color.append(float(color_err))
        loss_history_sil.append(float(sil_err))

        # Every 10 iterations, print the current values of the losses.
        if iteration % 10 == 0:
            print(
                f'Iteration {iteration:05d}:'
                + f' loss color = {float(color_err):1.2e}'
                + f' loss silhouette = {float(sil_err):1.2e}'
                + f' loss kl = {float(kl_err):1.2e}'
                + f' kl_factor = {kl_factor:1.3e}'
            )

        # Take the optimization step.
        loss.backward()
        optimizer.step()

        # TYXE comment: anneal the kl rate
        if iteration >= zero_kl_iters:
            kl_factor = min(max_kl_factor, kl_factor + kl_annealing_rate)

        # Visualize the full renders every 100 iterations.
        if iteration % 1000 == 0:
            show_idx = torch.randperm(len(target_cameras))[:1]
            fig = show_full_render(
                neural_radiance_field,
                FoVPerspectiveCameras(
                    R=target_cameras.R[show_idx],
                    T=target_cameras.T[show_idx],
                    znear=target_cameras.znear[show_idx],
                    zfar=target_cameras.zfar[show_idx],
                    aspect_ratio=target_cameras.aspect_ratio[show_idx],
                    fov=target_cameras.fov[show_idx],
                    device=device,
                ),
                target_images[show_idx][0],
                target_silhouettes[show_idx][0],
                loss_history_color,
                loss_history_sil,
                renderer_grid,
                num_forward=test_samples
            )
            plt.savefig(f"nerf/full_render{iteration}.png")
            plt.close(fig)

    with torch.no_grad():
        rotating_nerf_frames, uncertainty_frames = generate_rotating_nerf(
            neural_radiance_field,
            target_cameras,
            renderer_grid,
            device,
            n_frames=3 * 5,
            num_forward=test_samples,
            save_visualization=save_visualization
        )

    for i, (img, uncertainty) in enumerate(zip(
            rotating_nerf_frames.clamp(0., 1.).cpu().numpy(), uncertainty_frames.cpu().numpy())):
        f, ax = plt.subplots(figsize=(1.625, 1.625))
        f.subplots_adjust(0, 0, 1, 1)
        ax.imshow(img)
        ax.set_axis_off()
        f.savefig(f"nerf/final_image{i}.jpg", bbox_inches="tight", pad_inches=0)
        plt.close(f)

        f, ax = plt.subplots(figsize=(1.625, 1.625))
        f.subplots_adjust(0, 0, 1, 1)
        ax.imshow(uncertainty, cmap="hot", vmax=0.75 ** 0.5)
        ax.set_axis_off()
        f.savefig(f"nerf/final_uncertainty{i}.jpg", bbox_inches="tight", pad_inches=0)
        plt.close(f)

    if save_state_dict is not None:
        if inference != "ml":
            raise ValueError("Saving the state dict is only available for ml inference for now.")
        state_dict = dict(neural_radiance_field.named_pytorch_parameters(dummy_data))
        torch.save(state_dict, save_state_dict)

    test_cameras, test_images, test_silhouettes = generate_cow_renders(
        num_views=10, azimuth_low=90, azimuth_high=180)

    del renderer_mc
    del target_cameras
    del target_images
    del target_silhouettes
    torch.cuda.empty_cache()

    test_cameras = test_cameras.to(device)
    test_images = test_images.to(device)
    test_silhouettes = test_silhouettes.to(device)

    # TODO remove duplication from training code for test error
    with torch.no_grad():
        sil_err = 0.
        color_err = 0.
        for i in range(len(test_cameras)):
            batch_idx = [i]

            # Sample the minibatch of cameras.
            batch_cameras = FoVPerspectiveCameras(
                R=test_cameras.R[batch_idx],
                T=test_cameras.T[batch_idx],
                znear=test_cameras.znear[batch_idx],
                zfar=test_cameras.zfar[batch_idx],
                aspect_ratio=test_cameras.aspect_ratio[batch_idx],
                fov=test_cameras.fov[batch_idx],
                device=device,
            )

            img_list, sils_list, sampled_rays_list,  = [], [], []
            for _ in range(test_samples):
                rendered_images_silhouettes, sampled_rays = renderer_grid(
                    cameras=batch_cameras,
                    volumetric_function=partial(batched_forward, net=neural_radiance_field)
                )
                imgs, sils = (
                    rendered_images_silhouettes.split([3, 1], dim=-1)
                )
                img_list.append(imgs)
                sils_list.append(sils)
                sampled_rays_list.append(sampled_rays.xys)

            assert sampled_rays_list[0].eq(torch.stack(sampled_rays_list)).all()

            rendered_images = torch.stack(img_list).mean(0)
            rendered_silhouettes = torch.stack(sils_list).mean(0)

            # Compute the silhoutte error as the mean huber
            # loss between the predicted masks and the
            # sampled target silhouettes.
            # TYXE comment: sampled_rays are always the same for renderer_grid
            silhouettes_at_rays = sample_images_at_mc_locs(
                test_silhouettes[batch_idx, ..., None],
                sampled_rays.xys
            )
            sil_err += huber(
                rendered_silhouettes,
                silhouettes_at_rays,
            ).abs().mean().item() / len(test_cameras)

            # Compute the color error as the mean huber
            # loss between the rendered colors and the
            # sampled target images.
            colors_at_rays = sample_images_at_mc_locs(
                test_images[batch_idx],
                sampled_rays.xys
            )
            color_err += huber(
                rendered_images,
                colors_at_rays,
            ).abs().mean().item() / len(test_cameras)

    print(f"Test error: sil={sil_err:1.3e}; col={color_err:1.3e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", choices=["ml", "map", "mean-field"], required=True)
    parser.add_argument("--n-iter", type=int, default=20000)
    parser.add_argument("--save-state-dict")
    parser.add_argument("--load-state-dict")
    parser.add_argument("--kl-annealing-iters", type=int, default=0)
    parser.add_argument("--zero-kl-iters", type=int, default=0)
    parser.add_argument("--max-kl-factor", type=float, default=1.)
    parser.add_argument("--init-scale", type=float, default=1e-2)
    parser.add_argument("--save-visualization", action="store_true")
    main(**vars(parser.parse_args()))
