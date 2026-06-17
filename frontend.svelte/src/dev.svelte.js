import { mountRokaSveltePreview } from './main.svelte.js'
import './style.css'

const target = document.getElementById('app')
const preview = mountRokaSveltePreview(target, { aspectRatio: '1:1' })

window.rokaSveltePreview = preview
