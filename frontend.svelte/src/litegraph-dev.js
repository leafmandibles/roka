import { mountRokaSveltePreview } from './main.svelte.js'
import './style.css'
import '/@fs/home/nyika/.btca/agent/sandbox/litegraph.js/css/litegraph.css'

const { LiteGraph, LGraph, LGraphCanvas } = window

const DEFAULT_JSON = `{
  "compositional_deconstruction": {
    "elements": [
      { "type": "obj", "bbox": [140, 420, 980, 780], "desc": "fuel station canopy" },
      { "type": "text", "bbox": [60, 510, 135, 705], "text": "SHELL", "desc": "green petrol station sign text" }
    ]
  }
}`

const app = document.getElementById('app')
app.innerHTML = `
  <div class="litegraph-dev-shell">
    <canvas id="litegraph-canvas" width="1100" height="720"></canvas>
    <div id="litegraph-dom-layer"></div>
  </div>
`

const canvas = document.getElementById('litegraph-canvas')
const domLayer = document.getElementById('litegraph-dom-layer')
const graph = new LGraph()
const graphCanvas = new LGraphCanvas(canvas, graph)

function RokaLayoutNode() {
  this.addInput('ideogram_json', 'string')
  this.addOutput('ideogram_json', 'string')
  this.properties = { ideogram_json: DEFAULT_JSON, aspect_ratio: '1:1' }
  this.size = [420, 560]
}

RokaLayoutNode.title = 'RK Layout LiteGraph Harness'
RokaLayoutNode.prototype.onExecute = function () {
  this.setOutputData(0, this.properties.ideogram_json)
}
RokaLayoutNode.prototype.onDrawForeground = function (ctx) {
  ctx.fillStyle = '#87e8aa'
  ctx.font = '12px sans-serif'
  ctx.fillText('Svelte DOM widget mounted over this LiteGraph node', 12, 34)
}

LiteGraph.registerNodeType('roka/layout_harness', RokaLayoutNode)

const node = LiteGraph.createNode('roka/layout_harness')
node.pos = [120, 80]
graph.add(node)
graph.start()

const widgetEl = document.createElement('div')
widgetEl.className = 'litegraph-dev-widget'
domLayer.appendChild(widgetEl)

const preview = mountRokaSveltePreview(widgetEl, {
  ideogramJson: node.properties.ideogram_json,
  aspectRatio: node.properties.aspect_ratio,
  previewOnly: true,
  onIdeogramJsonChange(value) {
    console.warn('[RK LiteGraph Dev] onIdeogramJsonChange', { length: value?.length })
    node.properties.ideogram_json = value
  }
})

function graphToScreen(x, y) {
  const ds = graphCanvas.ds
  return [(x + ds.offset[0]) * ds.scale, (y + ds.offset[1]) * ds.scale]
}

function syncWidget() {
  const [x, y] = graphToScreen(node.pos[0] + 16, node.pos[1] + 58)
  const scale = graphCanvas.ds.scale || 1
  widgetEl.style.left = `${x}px`
  widgetEl.style.top = `${y}px`
  widgetEl.style.width = `${Math.max(260, (node.size[0] - 32) * scale)}px`
  widgetEl.style.height = `${Math.max(240, (node.size[1] - 78) * scale)}px`
  widgetEl.style.transformOrigin = '0 0'
}

const originalDraw = graphCanvas.draw.bind(graphCanvas)
graphCanvas.draw = function (...args) {
  const result = originalDraw(...args)
  syncWidget()
  return result
}

window.addEventListener('resize', () => {
  canvas.width = Math.min(window.innerWidth - 64, 1200)
  graphCanvas.resize()
  syncWidget()
})

syncWidget()
window.rokaLiteGraphDev = { graph, graphCanvas, node, preview }
console.warn('[RK LiteGraph Dev] loaded', window.rokaLiteGraphDev)
