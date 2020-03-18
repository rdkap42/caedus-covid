export default {
  data() {
    return {
      form: this.$form.createForm(this),
      submitStatus: null,
      statusTimeoutID: null
    }
  },

  mounted() {
    this.$nextTick(() => {
      // Disable submit button at the beginning.
      this.form.validateFields()
    })
  },

  methods: {
    hasErrors(fieldsError) {
      return Object.keys(fieldsError).some(field => fieldsError[field])
    },
    formError(field) {
      const { getFieldError, isFieldTouched } = this.form
      return isFieldTouched(field) && getFieldError(field)
    },
    markError() {
      // Set status to error
      this.submitStatus = 'error'

      // Clear timeout if it exists already
      if (this.statusTimeoutID) {
        clearTimeout(this.statusTimeoutID)
      }

      // Reset status after 4 seconds
      this.statusTimeoutID = setTimeout(() => {
        this.submitStatus = null
      }, 4000)
    },
    submit() {
      // Set status to pending
      this.submitStatus = 'pending'

      // Validate form
      this.form.validateFields(async (errors, values) => {
        if (!errors) {
          try {
            const response = await this.performSubmit(values)

            // Handle success
            this.submitStatus = 'success'
            if (typeof this.submitSuccess === 'function') {
              this.submitSuccess(response)
            }
          } catch (error) {
            // Handle errors
            this.markError()
            if (typeof this.submitError === 'function') this.submitError(error)
          }
        }
      })
    }
  }
}
