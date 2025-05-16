<template>
  <div id="mainDiv">
    <img :src="image_url" id="frameViewer"/>
    <div>
      <p class="ml-2">Ultimo rilevamento: {{ last_prediction }}</p>
    </div>
  </div>
</template>

<style scoped>
#frameViewer {
  width: 50vw;
  object-fit: cover;
}

#mainDiv {
  margin: auto;
  padding: 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
</style>

<script setup>
import Checkbox from 'primevue/checkbox';
import {onMounted, ref, onUnmounted} from 'vue'
import axios from 'axios'

const encoder = new TextEncoder("utf-8");
const last_prediction = ref("Elaborazione...");
const image_url = window.location.protocol+"//"+window.location.host+"/video_feed"

async function updateLabel() {
    const url = window.location.protocol+"//"+window.location.host+"/get_label"
    try {
        const response = await axios.get(url  , {
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        });
        last_prediction.value = response.data;
    } catch (error) {
        console.error('Error fetching label:', error);
    }
}

onMounted(() => {
  setInterval(updateLabel, 1000);
})

onUnmounted(()=>{

})
</script>