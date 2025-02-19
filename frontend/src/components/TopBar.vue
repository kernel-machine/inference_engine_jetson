<template>
  <div class="top-bar">
    <div style="display: flex; align-items: center;">
      <img src="/EV2.png" alt="Logo" class="logo"/>
    </div>
    <div>
      <a href="https://alcorlab.diag.uniroma1.it/projects/ev2">
        <Button>Info</Button>
      </a>
    </div>
  </div>
</template>

<script>
export default {
  name: 'TopBar',
}
</script>

<style scoped>
.top-bar {
  background-color: #ffffff;
  color: #333333;
  padding: 10px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.logo {
  width: auto;
  height: 60px;
}

.text {
  margin-left: 1em;
  font-size: 1.2rem;
}

button{
  background-color: #fbcc2a;
  color: black;
  border-color: black;
}
</style>
