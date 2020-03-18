<template lang="pug">
  .error(:class="'error_' + error.statusCode")
    h1(v-if="error.statusCode === 404") Page Not Found
    h1(v-if="error.statusCode === 500") Server Error

    h2 ({{ error.statusCode }})

    .actions
      nuxt-link.button(:to="{ name: 'index' }")
        a-icon(type="arrow-left")
        | Go back
      a.button(
        v-if="error.statusCode === 500"
        @click="$router.go()"
      )
        a-icon(type="reload")
        | Retry
      a.button(@click="$router.back()")
        a-icon(type="home")
        | Go home
</template>

<script>
export default {
  name: 'Error',
  props: {
    error: {
      type: Object,
      required: true
    }
  },
  layout: 'default'
}
</script>

<style scoped lang="scss">
$foregroundColor: #000;

.error {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20vh 2rem 2rem 2rem;
  text-align: center;
}

h1 {
  color: inherit;
  font-size: 5rem;
  margin: 0;
  line-height: 1;
}

h2 {
  color: inherit;
  font-size: 3rem;
  opacity: 0.8;
  margin: 0;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  margin-top: 4rem;
}

.button {
  margin: 0.5rem;
  padding: 0.5rem 1rem;
  display: block;
  border-style: solid;
  border-color: rgba($foregroundColor, 0.5);
  border-width: 2px;
  border-radius: 2px;
  color: inherit;
  font-size: 1.4rem;
  transition: background-color 0.3s;

  &:hover {
    background-color: rgba($foregroundColor, 0.2);
    text-decoration: none;
  }

  &:active {
    background-color: rgba($foregroundColor, 0.3);
  }

  .anticon {
    margin-right: 0.5rem;
  }
}
</style>
