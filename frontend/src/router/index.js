import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router)

export function createRouter(ssrContext, createDefaultRouter) {
  const defaultRouter = createDefaultRouter(ssrContext)
  return new Router({
    ...defaultRouter.options,
    routes: fixRoutes(defaultRouter.options.routes)
  })
}

function fixRoutes(defaultRoutes) {
  // default routes that come from `pages/`
  return defaultRoutes.map(route => {
    route.props = true
    return route
  })
}
