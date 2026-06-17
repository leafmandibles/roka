<script>
  import Bbox from './Bbox.svelte'

  let { aspectRatio = '1:1', elements = [], onBoxMove = () => {} } = $props()

  let fitEl
  let fitSize = $state({ width: 0, height: 0 })

  const clamp = (value) => Math.max(0, Math.min(1000, Number(value) || 0))

  let ratio = $derived.by(() => {
    const [w, h] = String(aspectRatio || '1:1').split(':').map(Number)
    return w > 0 && h > 0 ? w / h : 1
  })

  let stageSize = $derived.by(() => {
    const width = Math.max(0, fitSize.width)
    const height = Math.max(0, fitSize.height)
    if (!width || !height) return { width: '100%', height: '100%' }

    const availableRatio = width / height
    if (availableRatio > ratio) {
      const stageHeight = height
      return { width: `${stageHeight * ratio}px`, height: `${stageHeight}px` }
    }

    const stageWidth = width
    return { width: `${stageWidth}px`, height: `${stageWidth / ratio}px` }
  })

  let boxViews = $derived((Array.isArray(elements) ? elements : []).map((element, index) => {
    const [y1, x1, y2, x2] = Array.isArray(element?.bbox) ? element.bbox.map(clamp) : [0, 0, 0, 0]
    return {
      element,
      index,
      top: Math.min(y1, y2) / 10,
      left: Math.min(x1, x2) / 10,
      width: Math.abs(x2 - x1) / 10,
      height: Math.abs(y2 - y1) / 10,
      x1: Math.min(x1, x2),
      y1: Math.min(y1, y2),
      x2: Math.max(x1, x2),
      y2: Math.max(y1, y2),
      label: element.text || element.desc || element.type || `#${index + 1}`
    }
  }))

  let calculatedAspectRatio = $derived.by(() => {
    const boxes = boxViews.filter((box) => box.width > 0 && box.height > 0)
    if (!boxes.length) return ''
    const minX = Math.min(...boxes.map((box) => box.x1))
    const minY = Math.min(...boxes.map((box) => box.y1))
    const maxX = Math.max(...boxes.map((box) => box.x2))
    const maxY = Math.max(...boxes.map((box) => box.y2))
    const width = maxX - minX
    const height = maxY - minY
    if (width <= 0 || height <= 0) return ''
    const candidates = ['1:1', '4:3', '3:4', '16:9', '9:16', '3:2', '2:3', '5:4', '4:5', '21:9']
    const actual = width / height
    return candidates.reduce((best, label) => {
      const [w, h] = label.split(':').map(Number)
      const score = Math.abs(Math.log(actual / (w / h)))
      return score < best.score ? { label, score } : best
    }, { label: '1:1', score: Infinity }).label
  })

  $effect(() => {
    if (!fitEl) return
    const update = () => {
      fitSize = { width: fitEl.clientWidth, height: fitEl.clientHeight }
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(fitEl)
    return () => observer.disconnect()
  })
</script>

<div class="bbox-preview__stage-wrap">
  <div class="bbox-preview__meta">
    <strong>BBox preview{calculatedAspectRatio ? ` (${calculatedAspectRatio})` : ''}</strong>
    <span>{boxViews.length} element{boxViews.length === 1 ? '' : 's'}</span>
  </div>

  <div class="bbox-preview__fit" bind:this={fitEl}>
    <div
      class="bbox-preview__stage"
      style:width={stageSize.width}
      style:height={stageSize.height}
    >
      {#each boxViews as box}
        <Bbox
          left={box.left}
          top={box.top}
          width={box.width}
          height={box.height}
          label={box.label}
          onmove={(position) => onBoxMove(box.index, position)}
        />
      {/each}
    </div>
  </div>
</div>
