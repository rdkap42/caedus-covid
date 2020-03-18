<template lang="pug">
  a-layout-header
    nuxt-link(to='/').branding
      .title Caedus Covid

    a-menu(
      theme='dark'
      mode='horizontal'
      :default-selected-keys="['2']"
      :style="{ lineHeight: '64px' }"
    )
      a-menu-item(key='test')
        nuxt-link(:to="{ name: 'test' }")
          | Menu item
</template>

<script>
export default {
  name: 'PageHeader',
  methods: {}
}
</script>

<style lang="scss" scoped>
.ant-layout-header {
  padding: 0 2rem;
  position: sticky;
  top: 0;
  z-index: 1001;
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
  background-color: #222;
  color: #fff;

  .ant-menu {
    background-color: #222;
    color: #fff;
    font-size: 1.2rem;

    .ant-menu-item-selected {
      background-color: rgba(0, 0, 0, 0.2);
    }
  }

  .branding {
    height: 100%;
    padding: 0.5rem 0;
    display: flex;
    flex-direction: row;
    align-items: center;
    color: #fff;

    .title {
      font-size: 1.8rem;
    }
  }
}

@media (max-width: 800px) {
  .ant-layout-header {
    justify-content: center;
  }
}
</style>
