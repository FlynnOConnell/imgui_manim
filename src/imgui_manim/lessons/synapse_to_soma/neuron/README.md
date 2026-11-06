# Reference neuron and synapse

The panels show one mouse visual cortex pyramidal cell and the synapse on one of its spines. `cell.png`, `branch.png`,
`spine.png`, `soma.png` and `neuron.json` are copies of the panels in imgui_ophys's Subcellular page, kept here because
`path.npz` is registered to `cell.png` pixel for pixel. `synapse.png` is an electron micrograph of Spine 1's synapse.
`path.npz` gives every pixel of `cell.png` its path distance from the soma and from Spine 1, and holds the outlines of
the bouton and the spine in `synapse.png`.

The cell and the synapse are real, and the voltages drawn on them are simulated. The model is not fitted to this cell.

`scripts/synapse_to_soma/export_neuron.py` writes this folder.

## Source

Segment 864691135572530981 of the MICrONS cubic millimeter dataset, from the static meshes `seg_m1300` at
`precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300`, and the electron micrographs at
`precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em`. The bouton on Spine 1 is
segment 864691135502216470. www.microns-explorer.org

Cite it as:

> The MICrONS Consortium, J. Alexander Bae, Mahaly Baptiste, Maya R. Baptiste, Caitlyn A. Bishop, Agnes L. Bodor, Derrick
> Brittain, et al. 2025. "Functional Connectomics Spanning Multiple Areas of Mouse Visual Cortex." Nature 640 (8058):
> 435-47. https://doi.org/10.1038/s41586-025-08790-w

MICrONS asks that its content is never displayed without this citation, and that a website using it links to
www.microns-explorer.org. Keep the credit next to these images wherever they appear.
(https://www.microns-explorer.org/citation-policy)

## License

The source data are licensed by the MICrONS Consortium under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode), which comes with its disclaimer of warranties.
These images are adapted from them under the same terms. This license covers the images, `neuron.json` and `path.npz` in
this folder; the repository's MIT license covers its code.

## Changes from the source

- The surface mesh is drawn as a lit, shaded image with a z-buffer and ambient occlusion, on a transparent background.
- The whole-cell panel uses the level-of-detail 1 mesh with its surface pushed out 0.3 um, because true branches are
  about 1 px wide at that scale.
- `synapse.png` is one 8 nm plane, contrast stretched and enlarged 3 times. The outlines come from the segmentation.
- Path distances are shortest paths over 1 um occupied voxels and are approximate.

## Limits

The meshes have no compartment labels, so axon and dendrite are not told apart: a voltage painted by path distance also
runs along the axon. The cell's proofreading status was not checked, because that table needs a CAVE account, and static
meshes can hold false gaps and merges. That the synapse is excitatory is inferred from its place on a spine head.
