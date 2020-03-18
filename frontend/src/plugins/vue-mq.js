import Vue from 'vue'
import VueMq from 'vue-mq'

Vue.use(VueMq, {
  breakpoints: {
    tiny: 800,
    small: 1000,
    medium: 1200,
    desktop_small: 1600,
    desktop: 2200,
    desktop_large: Infinity
  },
  defaultBreakpoint: 'small'
})
