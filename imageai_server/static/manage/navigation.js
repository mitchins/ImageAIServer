// Shared navigation component for ImageAI UI pages
// Automatically injects a consistent navigation bar at the top of pages

(function() {
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initNavigation);
    } else {
        initNavigation();
    }
    
    function initNavigation() {
        const navHTML = `
            <nav class="imageai-nav">
                <div class="nav-container">
                    <div class="nav-brand">
                        <a href="/" class="brand-link"><img src="/static/icon.png" alt="ImageAI" class="nav-icon"> ImageAI</a>
                    </div>
                    <div class="nav-links">
                        <a href="/" class="nav-link" title="Home">🏠 Home</a>
                        <a href="/manage/ui/" class="nav-link" title="Model Management">🌐 Models</a>
                        <a href="/manage/ui/vision-test.html" class="nav-link" title="Vision & Face Testing">🎯 Test</a>
                        <a href="/docs" class="nav-link" title="API Documentation">📚 Docs</a>
                    </div>
                </div>
            </nav>
        `;
        
        // Insert navigation at the beginning of body (only if not already present)
        if (!document.querySelector('.imageai-nav')) {
            document.body.insertAdjacentHTML('afterbegin', navHTML);
            document.body.classList.add('has-nav');
        }
        
        // Highlight current page
        highlightCurrentPage();
    }
    
    function highlightCurrentPage() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href === currentPath || 
                (href === '/manage/ui/' && currentPath.includes('/manage/ui/')) ||
                (href === '/docs' && currentPath.includes('/docs'))) {
                link.classList.add('active');
            }
        });
    }
})();
