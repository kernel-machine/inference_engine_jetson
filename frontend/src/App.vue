<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import TopBar from './components/TopBar.vue'
</script>

<template>
  <header>
    <TopBar></TopBar>
<!--    <div class="wrapper">-->
<!--      <nav>-->
<!--        <RouterLink to="/">Home</RouterLink>-->
<!--         <RouterLink to="/about">About</RouterLink> -->
<!--      </nav>-->
<!--    </div>-->
  </header>

  <RouterView />
</template>

<style scoped>

</style>
