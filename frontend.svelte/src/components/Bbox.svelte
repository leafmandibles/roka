<script>
  let {
    left = 0,
    top = 0,
    width = 0,
    height = 0,
    label = '',
    onmove = () => {}
  } = $props()

  let drag = $state(null)
  let offset = $state({ left: 0, top: 0 })

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value))

  let displayLeft = $derived(clamp(left + offset.left, 0, 100 - width))
  let displayTop = $derived(clamp(top + offset.top, 0, 100 - height))

  function dragListener(node) {
    console.warn('[RK Bbox] attach fired native listener drag-attach-v1', { label, node })
    node.addEventListener('pointerdown', handlePointerDown, { capture: true })
    return () => {
      node.removeEventListener('pointerdown', handlePointerDown, { capture: true })
    }
  }

  function handlePointerDown(event) {
    console.log('[RK Bbox] pointerdown', {
      label,
      button: event.button,
      pointerId: event.pointerId,
      target: event.target,
      currentTarget: event.currentTarget
    })

    if (drag || event.button !== 0) return

    const pointerId = event.pointerId ?? 'mouse'
    const stage = event.currentTarget.parentElement
    const stageRect = stage?.getBoundingClientRect()
    console.log('[RK Bbox] stage rect', { label, stage, stageRect })
    if (!stageRect?.width || !stageRect?.height) return

    event.preventDefault()
    event.stopPropagation()
    drag = {
      pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startOffset: offset,
      stageRect
    }
    window.addEventListener('pointermove', handleWindowPointerMove)
    window.addEventListener('pointerup', handleWindowPointerUp)
    window.addEventListener('pointercancel', handleWindowPointerUp)
  }

  function handleWindowPointerMove(event) {
    const pointerId = event.pointerId ?? 'mouse'
    if (!drag || pointerId !== drag.pointerId) return

    event.preventDefault()
    console.log('[RK Bbox] pointermove', {
      label,
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY
    })

    offset = {
      left: drag.startOffset.left + ((event.clientX - drag.startX) / drag.stageRect.width) * 100,
      top: drag.startOffset.top + ((event.clientY - drag.startY) / drag.stageRect.height) * 100
    }
  }

  function handleWindowPointerUp(event) {
    const pointerId = event.pointerId ?? 'mouse'
    if (!drag || pointerId !== drag.pointerId) return

    window.removeEventListener('pointermove', handleWindowPointerMove)
    window.removeEventListener('pointerup', handleWindowPointerUp)
    window.removeEventListener('pointercancel', handleWindowPointerUp)
    console.log('[RK Bbox] pointerup commit', {
      label,
      left: displayLeft,
      top: displayTop,
      width,
      height
    })
    onmove({ left: displayLeft, top: displayTop, width, height })
    offset = { left: 0, top: 0 }
    drag = null
  }
</script>

<div
  class="bbox-preview__box"
  {@attach dragListener}
  class:bbox-preview__box--dragging={drag}
  style:left={`${displayLeft}%`}
  style:top={`${displayTop}%`}
  style:width={`${width}%`}
  style:height={`${height}%`}
>
  <span>{label}</span>
</div>
