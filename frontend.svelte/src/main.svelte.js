import { mount, unmount } from 'svelte'
import App from './App.svelte'
import './style.css'

const DEFAULT_JSON = `{
  "compositional_deconstruction": {
    "elements": [
      {
        "type": "obj",
        "bbox": [140, 420, 980, 780],
        "desc": "fuel station canopy"
      },
      {
        "type": "text",
        "bbox": [60, 510, 135, 705],
        "text": "SHELL",
        "desc": "green petrol station sign text"
      }
    ]
  }
}`

const DEFAULT_SCENEGRAPH_JSON = `[
  {
    "id": 0,
    "label": "foreground",
    "parent_id": null
  },
  {
    "id": 1,
    "label": "masked figure",
    "parent_id": 0,
    "bbox": [180, 80, 760, 950],
    "source_ids": [4]
  },
  {
    "id": 2,
    "label": "glowing orange orb",
    "parent_id": 0,
    "bbox": [460, 430, 620, 590],
    "source_ids": [7]
  },
  {
    "id": 3,
    "label": "midground",
    "parent_id": null
  },
  {
    "id": 4,
    "label": "exotic sports car hood",
    "parent_id": 3,
    "bbox": [310, 10, 930, 650],
    "source_ids": [2]
  },
  {
    "id": 5,
    "label": "background",
    "parent_id": null
  },
  {
    "id": 6,
    "label": "gas station canopy and pumps",
    "parent_id": 5,
    "bbox": [0, 0, 390, 1000],
    "source_ids": [0, 1]
  }
]`

function isValidAspectRatio(value) {
  const match = String(value || '').trim().match(/^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$/)
  return !!match && Number(match[1]) > 0 && Number(match[2]) > 0
}

function extractElements(text) {
  const parsed = JSON.parse(text || '[]')
  if (Array.isArray(parsed)) return parsed
  const elements = parsed?.compositional_deconstruction?.elements
  return Array.isArray(elements) ? elements : []
}

function sceneGraphNodes(parsed) {
  if (Array.isArray(parsed)) return parsed
  if (Array.isArray(parsed?.nodes)) return parsed.nodes
  if (Array.isArray(parsed?.scenegraph)) return parsed.scenegraph
  if (Array.isArray(parsed?.scene_graph)) return parsed.scene_graph
  return []
}

function extractSceneGraphElements(text) {
  const parsed = JSON.parse(text || '[]')
  return sceneGraphNodes(parsed)
    .map((node, nodeIndex) => ({ node, nodeIndex }))
    .filter(({ node }) => Array.isArray(node?.bbox) && node.bbox.length === 4)
    .map(({ node, nodeIndex }) => {
      const [x1, y1, x2, y2] = node.bbox.map((value) => Number(value) || 0)
      return {
        ...node,
        _sceneGraphNodeIndex: nodeIndex,
        type: node.type || 'obj',
        desc: node.desc || node.label || node.type || `#${node.id ?? nodeIndex + 1}`,
        bbox: [y1, x1, y2, x2]
      }
    })
}

function patchSceneGraphBbox(text, previewIndex, position) {
  const parsed = JSON.parse(text || '[]')
  const nodes = sceneGraphNodes(parsed)
  const bboxNodeIndexes = nodes
    .map((node, nodeIndex) => ({ node, nodeIndex }))
    .filter(({ node }) => Array.isArray(node?.bbox) && node.bbox.length === 4)
    .map(({ nodeIndex }) => nodeIndex)
  const nodeIndex = bboxNodeIndexes[previewIndex]
  if (nodeIndex == null || !nodes[nodeIndex]) return null

  const x1 = Math.round(position.left * 10)
  const y1 = Math.round(position.top * 10)
  const x2 = Math.round((position.left + position.width) * 10)
  const y2 = Math.round((position.top + position.height) * 10)
  nodes[nodeIndex] = { ...nodes[nodeIndex], bbox: [x1, y1, x2, y2] }
  return JSON.stringify(parsed, null, 2)
}

function createPreviewState({ ideogramJson, sceneGraphJson, aspectRatio = '1:1', mode = 'ideogram', onIdeogramJsonChange = () => {}, onSceneGraphJsonChange = () => {} } = {}) {
  const isSceneGraph = mode === 'scenegraph'
  const initialJson = String((isSceneGraph ? sceneGraphJson : ideogramJson) ?? (isSceneGraph ? DEFAULT_SCENEGRAPH_JSON : DEFAULT_JSON))
  const initialAspectRatio = String(aspectRatio || '1:1')
  let currentJson = $state(initialJson)
  let currentAspectRatio = $state(initialAspectRatio)
  let validAspectRatio = $state(isValidAspectRatio(initialAspectRatio) ? initialAspectRatio.trim() : '1:1')
  let validElements = $state([])
  let jsonError = $state('')

  function setIdeogramJson(value) {
    currentJson = String(value ?? '')
    try {
      validElements = isSceneGraph ? extractSceneGraphElements(currentJson) : extractElements(currentJson)
      jsonError = ''
    } catch (err) {
      jsonError = err.message || String(err)
    }
  }

  function setAspectRatio(value) {
    currentAspectRatio = String(value ?? '')
    if (isValidAspectRatio(currentAspectRatio)) validAspectRatio = currentAspectRatio.trim()
  }

  function moveElement(index, position) {
    console.log('[RK Layout] moveElement', { index, position, isSceneGraph })
    try {
      if (isSceneGraph) {
        const nextJson = patchSceneGraphBbox(currentJson, index, position)
        if (!nextJson) return
        console.log('[RK SceneGraphLayout] nextJson', nextJson)
        setIdeogramJson(nextJson)
        onSceneGraphJsonChange(nextJson)
        return
      }

      const parsed = JSON.parse(currentJson || '{}')
      const elements = Array.isArray(parsed)
        ? parsed
        : parsed?.compositional_deconstruction?.elements
      if (!Array.isArray(elements) || !elements[index]) return

      const y1 = Math.round(position.top * 10)
      const x1 = Math.round(position.left * 10)
      const y2 = Math.round((position.top + position.height) * 10)
      const x2 = Math.round((position.left + position.width) * 10)
      elements[index] = { ...elements[index], bbox: [y1, x1, y2, x2] }

      const nextJson = JSON.stringify(parsed, null, 2)
      console.log('[RK Layout] nextJson', nextJson)
      setIdeogramJson(nextJson)
      onIdeogramJsonChange(nextJson)
    } catch (err) {
      jsonError = err.message || String(err)
    }
  }

  setIdeogramJson(initialJson)

  return {
    get ideogramJson() { return currentJson },
    set ideogramJson(value) { setIdeogramJson(value) },
    get sceneGraphJson() { return currentJson },
    set sceneGraphJson(value) { setIdeogramJson(value) },
    get isSceneGraph() { return isSceneGraph },
    get inputLabel() { return isSceneGraph ? 'SceneGraph JSON' : 'Ideogram v4 JSON or elements array' },
    get outputLabel() { return isSceneGraph ? 'Output: SceneGraph JSON pass-through' : 'Output: Ideogram JSON pass-through' },
    get aspectRatio() { return currentAspectRatio },
    set aspectRatio(value) { setAspectRatio(value) },
    get validAspectRatio() { return validAspectRatio },
    get validElements() { return validElements },
    get jsonError() { return jsonError },
    setIdeogramJson,
    setAspectRatio,
    moveElement
  }
}

export function mountRokaSveltePreview(target, props = {}) {
  const state = createPreviewState(props)
  const app = mount(App, { target, props: { state, previewOnly: !!props.previewOnly } })

  return {
    updateIdeogramJson(value) { state.setIdeogramJson(value) },
    updateSceneGraphJson(value) { state.setIdeogramJson(value) },
    updateElementsText(value) { state.setIdeogramJson(value) },
    updateAspectRatio(value) { state.setAspectRatio(value) },
    destroy() { unmount(app) }
  }
}

if (typeof window !== 'undefined') {
  window.mountRokaSveltePreview = mountRokaSveltePreview
}
