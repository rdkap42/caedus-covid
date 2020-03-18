<template lang="pug">
  .index
    h1 Welcome to the app!
</template>

<script>
export default {
  name: 'Index',
  auth: false,
  head() {
    return {
      title: 'Home',
      titleTemplate: null,
      meta: [
        {
          hid: 'description',
          name: 'description',
          content: 'The home page'
        }
      ]
    }
  }
}
</script>

<style lang="scss" scoped>
.index {
  text-align: center;
}
</style>
