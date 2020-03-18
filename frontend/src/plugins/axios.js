export default function({ $axios, redirect }) {
  $axios.defaults.headers.common.Accept = 'application/json; version=1.0'
}
