![](./docs/images/logo.png)

## Intro

![Roka image understanding example](./docs/images/roka-intro-example.png)

Roka provides nodes Scene Graph Analysis.  Contains nodes that will take a set of segmentation masks and boxes and generates a scenegraph. The Scenegraph can be used for downstream tasks
 * Editing models (e.g Flux Klein, Qwen Image Edit) 
 * Debugging prompt comprehension 
 * Support Ideogram V4 prompting 



The nodes including SAM3-powered text segmentation, scene-graph extraction, and text post-processing helpers for prompt or metadata pipelines.

## Examples

The table below shows Ideogram V4 JSON-prompt renders produced from Roka scene-graph composition outputs. Rows that were replaced by the erroneous image safety-filter placeholder are intentionally omitted.

| Image | Composition style | Original image overlay | Rendered image overlay |
|---|---|---|---|
| Cave skylight | simple | <img src="./docs/images/examples/cave-simple-original-overlay.png" alt="Cave simple original overlay" width="320"> | <img src="./docs/images/examples/cave-simple-overlay.png" alt="Cave simple rendered overlay" width="320"> |
| Cave skylight | horizontal | <img src="./docs/images/examples/cave-horizontal-original-overlay.png" alt="Cave horizontal original overlay" width="320"> | <img src="./docs/images/examples/cave-horizontal-overlay.png" alt="Cave horizontal rendered overlay" width="320"> |
| Harbor panorama | simple | <img src="./docs/images/examples/harbor-simple-original-overlay.png" alt="Harbor simple original overlay" width="320"> | <img src="./docs/images/examples/harbor-simple-overlay.png" alt="Harbor simple rendered overlay" width="320"> |
| Railway station | simple | <img src="./docs/images/examples/station-simple-original-overlay.png" alt="Station simple original overlay" width="320"> | <img src="./docs/images/examples/station-simple-overlay.png" alt="Station simple rendered overlay" width="320"> |

## Prerequisites 

 * models :  models/sam3/sam3.pt 
 * uv (for your virtualenvs) 
 

## Workflows

Agent/operator note: for saved Datavelt/Kaleidoscope workflow reruns, see [`docs/how-to-invoke-datavelt-workflows.md`](./docs/how-to-invoke-datavelt-workflows.md).

### RK Segmentation

![RK Segmentation workflow](./docs/images/rk-segmentation-workflow.png)

Workflow file: [`workflows/rk_sam3_scenegraph_example_workflow.json`](./workflows/rk_sam3_scenegraph_example_workflow.json)

This workflow loads an image, scales it, runs **RK SAM3 Multi Text Segmentation** with a comma-separated text prompt, and previews the resulting segmentation visualization.

### Scene Graph

![Scene Graph workflow](./docs/images/scene-graph-workflow.png)

Workflow file: [`workflows/RK_SAM3_SceneGraph.json`](./workflows/RK_SAM3_SceneGraph.json)

This workflow extends SAM3 segmentation with **RK SAM3 Scene Graph** to convert detected masks and boxes into a simple object hierarchy. The ASCII output makes relationships easy to inspect or pass downstream.

```text
0
├── [0] sky
├── [1] ground
└── [2] person
    └── [3] person
```

### RK spaCy Filter

![RK spaCy Filter workflow](./docs/images/rk-spacy-filter-workflow.png)

Workflow file: [`workflows/workflow_scenegraph_ascii_tree.json`](./workflows/workflow_scenegraph_ascii_tree.json)

This workflow uses a Florence2 captioning pipeline, then passes the generated caption through **RK spaCy Filter**. The node extracts and filters parts of speech, such as nominal nouns, while allowing custom stop words or excluded terms. 

> It is useful for turning verbose captions into a noun list for segmentation

### Scene Graph Renderer

![Scene Graph Renderer workflow](./docs/images/scene-graph-renderer-workflow.png)

**RK Scene Graph Renderer** visualizes the scene graph back over the image/mask space. It renders the detected hierarchy and boxes so you can inspect whether the graph structure matches the source image before using it downstream.

### Scene Graph Reducer

![Scene Graph Reducer workflow](./docs/images/scenegraph-reducer-composer-workflow.png)

![Scene Graph Reducer overlay](./docs/images/scenegraph-reducer-overlay.png)

**RK SceneGraphReducer** trims a scene graph to a configurable depth, making large detections easier to compose, render, or pass into downstream prompt-building nodes. The reduced graph can then be visualized with **RK SceneOverlay** to confirm the kept objects and spatial labels.
