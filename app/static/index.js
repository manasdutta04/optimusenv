document.addEventListener("DOMContentLoaded", () => {
    // Generate background particles
    const container = document.getElementById('particle-container');
    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Random properties
        const size = Math.random() * 5 + 2;
        const left = Math.random() * 100;
        const delay = Math.random() * 10;
        const duration = Math.random() * 10 + 10;
        
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${left}%`;
        particle.style.bottom = `-10px`;
        particle.style.animationDelay = `${delay}s`;
        particle.style.animationDuration = `${duration}s`;
        
        container.appendChild(particle);
    }

    // Fetch health data to populate stat cards
    fetch('/health')
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            document.getElementById('env-name').textContent = data.env;
            document.getElementById('env-version').textContent = 'v' + data.version;
            document.getElementById('env-status').textContent = 'Active';
            document.getElementById('env-status').style.color = 'var(--success)';
            
            const badge = document.getElementById('app-status');
            badge.innerHTML = '<span class="pulse-dot"></span> System Connected';
            badge.style.color = 'var(--success)';
            badge.style.background = 'rgba(16, 185, 129, 0.1)';
            badge.style.borderColor = 'rgba(16, 185, 129, 0.2)';
        })
        .catch(error => {
            console.error('Error fetching health status:', error);
            document.getElementById('env-name').textContent = 'OptimusEnv';
            document.getElementById('env-version').textContent = 'Unknown';
            document.getElementById('env-status').textContent = 'Offline';
            document.getElementById('env-status').style.color = '#ef4444';
            
            const badge = document.getElementById('app-status');
            badge.innerHTML = '<span class="pulse-dot" style="background:#ef4444;box-shadow:0 0 10px #ef4444"></span> System Offline';
            badge.style.color = '#ef4444';
            badge.style.background = 'rgba(239, 68, 68, 0.1)';
            badge.style.borderColor = 'rgba(239, 68, 68, 0.2)';
        });
});
