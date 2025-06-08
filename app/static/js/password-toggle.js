/**
 * Password visibility toggle functionality
 * Allows users to show/hide password field contents
 */
document.addEventListener('DOMContentLoaded', function() {
    // Get all password toggle buttons
    const toggleButtons = document.querySelectorAll('.password-toggle-btn');
    
    // Add click event listener to each button
    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Get the password input field
            const passwordField = document.getElementById(this.getAttribute('data-password-field'));
            
            // Toggle password visibility
            if (passwordField.type === 'password') {
                passwordField.type = 'text';
                this.innerHTML = '<i class="fas fa-eye"></i>';
                this.setAttribute('title', 'Hide password');
            } else {
                passwordField.type = 'password';
                this.innerHTML = '<i class="fas fa-eye-slash"></i>';
                this.setAttribute('title', 'Show password');
            }
        });
    });
});
