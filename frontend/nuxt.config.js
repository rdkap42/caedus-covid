const path = require('path')
const serveStatic = require('serve-static')

const srcDir = 'src/'

module.exports = {
  mode: 'universal',
  modern: true,
  srcDir,

  /*
   ** Headers of the page
   */
  head: {
    title: 'Caedus Covid',
    titleTemplate: '%s - Caedus Covid',
    meta: [
      { charset: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      {
        hid: 'description',
        name: 'description',
        content: ''
      }
    ],
    link: [{ rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }]
  },

  /*
   ** Customize the progress-bar color
   */
  loading: { color: 'green' },

  /*
   ** Global CSS
   */
  css: [{ src: 'ant-design-vue/dist/antd.less', lang: 'less' }],

  /*
   ** Plugins to load before mounting the App
   */
  plugins: [
    '@/plugins/ant-design-vue',
    '@/plugins/axios',
    '@/plugins/vue-mq',
    '@/plugins/vue2-filters'
  ],

  /*
   ** Server middleware
   */
  serverMiddleware: [
    'redirect-ssl'
  ],

  /*
   ** Nuxt.js modules
   */
  modules: [
    '@bazzite/nuxt-optimized-images',
    '@nuxtjs/axios',
    '@nuxtjs/robots',
    [
      '@nuxtjs/router',
      {
        path: path.resolve(srcDir, 'router/'),
        fileName: 'index.js',
        keepDefaultRouter: true
      }
    ],
    '@nuxtjs/svg-sprite',
    'nuxt-helmet',
    'nuxt-purgecss',

    // Keep sitemap at the end
    '@nuxtjs/sitemap'
  ],

  /*
   ** nuxt-optimized-images configuration
   */
  optimizedImages: {
    optimizeImages: true
  },

  robots: {
    UserAgent: '*',
    // Don't index any pages
    Disallow: '/'
  },

  svgSprite: {
    elementClass: 'svg-icon'
  },

  helmet: {
    contentSecurityPolicy: {
      directives: {
        connectSrc: [
          "'self'",
          new URL(process.env.API_URL).origin
        ],
        defaultSrc: ["'none'"],
        fontSrc: [
          "'self'",
          'data:'
        ],
        frameSrc: ["'none'"],
        imgSrc: [
          "'self'",
          'data:'
        ],
        objectSrc: ["'none'"],
        scriptSrc: [
          "'self'",
          "'unsafe-inline'",
          "'unsafe-eval'"
        ],
        styleSrc: [
          "'self'",
          "'unsafe-inline'"
        ]
      }
    },
    hsts: {
      includeSubDomains: false
    }
  },

  sitemap: {
    hostname: '',
    gzip: true,
    exclude: []
  },

  /*
   ** Build configuration
   */
  build: {
    loaders: {
      less: {
        javascriptEnabled: true
      }
    },
    babel: {
      plugins: [
        [
          'import',
          { libraryName: 'ant-design-vue', libraryDirectory: 'es', style: true }
        ]
      ]
    },

    /*
     ** You can extend webpack config here
     */
    extend(config, ctx) {
      // Run ESLint on save
      if (ctx.isDev && ctx.isClient) {
        config.module.rules.push({
          enforce: 'pre',
          test: /\.(js|vue)$/,
          loader: 'eslint-loader',
          exclude: /(node_modules)/
        })
      }

      // Add markdown support
      config.module.rules.push({
        test: /\.md$/,
        use: 'raw-loader'
      })
    }
  }
}
