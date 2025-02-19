<template>
  <div id="mainDiv">
    <img src="http://127.0.0.1:5000/video_feed" id="frameViewer"/>
    <div>
      <p class="ml-2">Ultimo rilevamento: {{ last_prediction }}</p>
    </div>
  </div>
</template>

<style scoped>
#frameViewer {
  flex:1;
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

async function updateLabel() {
    try {
        const response = await axios.get('http://127.0.0.1:5000/get_label', {
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