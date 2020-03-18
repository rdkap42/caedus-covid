<template lang="pug">
  a-layout.main-layout
    page-header
    a-layout-content
      main(role="main")
        nuxt
    page-footer
</template>

<script>
import PageHeader from '~/components/layout/PageHeader'
import PageFooter from '~/components/layout/PageFooter'

export default {
  components: {
    PageHeader,
    PageFooter
  }
}
</script>

<style lang="scss">
:root {
  font-size: 16px;
}

html {
  font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI',
    Roboto, 'Helvetica Neue', Arial, sans-serif;
  font-size: 16px;
  word-spacing: 1px;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
  -moz-osx-font-smoothing: grayscale;
  -webkit-font-smoothing: antialiased;
  box-sizing: border-box;
  scrollbar-color: #2dbab6 transparent;
}

*,
*:before,
*:after {
  box-sizing: border-box;
}

.ant-layout.main-layout {
  min-height: 100vh;
}
</style>
