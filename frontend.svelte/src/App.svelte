<script>
  import BboxPreview from './components/BboxPreview.svelte'

  let { state, previewOnly = false } = $props()
</script>

{#if previewOnly}
  {#if state.jsonError}
    <p class="bbox-preview__error">{state.jsonError}</p>
  {/if}
  <BboxPreview aspectRatio={state.validAspectRatio} elements={state.validElements} onBoxMove={state.moveElement} />
{:else}
  <section class="layout-preview">
    <div class="layout-preview__controls">
      <label>
        <span>Aspect ratio string</span>
        <input bind:value={state.aspectRatio} placeholder="1:1" spellcheck="false" />
      </label>

      <label>
        <span>{state.inputLabel}</span>
        <textarea bind:value={state.ideogramJson} spellcheck="false"></textarea>
      </label>
    </div>

    <div>
      {#if state.jsonError}
        <p class="bbox-preview__error">{state.jsonError}</p>
      {/if}

      <BboxPreview aspectRatio={state.validAspectRatio} elements={state.validElements} onBoxMove={state.moveElement} />

      <div class="bbox-preview__output">
        <div>{state.outputLabel}</div>
        <pre>{state.ideogramJson}</pre>
      </div>
    </div>
  </section>
{/if}
