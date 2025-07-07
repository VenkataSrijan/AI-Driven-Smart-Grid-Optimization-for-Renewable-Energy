document.addEventListener('DOMContentLoaded', () => {
    // Initialize Three.js for 3D background elements
    initThreeJS();
    
    // Elements for single action
    const getActionBtn = document.getElementById('getActionBtn');
    const batteryChargeInput = document.getElementById('batteryCharge');
    const netEnergyInput = document.getElementById('netEnergy');
    const avgFutureNetEnergyInput = document.getElementById('avgFutureNetEnergy');
    const timeStepInput = document.getElementById('timeStep');

    const actionDisplay = document.getElementById('actionDisplay');
    const descriptionDisplay = document.getElementById('descriptionDisplay');
    const qValuesDisplay = document.getElementById('qValuesDisplay');

    // Elements for simulation
    const runSimulationBtn = document.getElementById('runSimulationBtn');
    const simulationStatus = document.getElementById('simulationStatus');

    // Chart instances
    let batteryChartInstance = null;
    let energyFlowChartInstance = null;
    let lossChartInstance = null;

    // --- Three.js Initialization ---
    function initThreeJS() {
        const container = document.getElementById('threejs-container');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);
        
        // Position camera
        camera.position.z = 30;
        
        // Add floating geometric elements
        const geometry = new THREE.IcosahedronGeometry(1, 0);
        const material = new THREE.MeshBasicMaterial({ 
            color: 0x4361ee,
            transparent: true,
            opacity: 0.1,
            wireframe: true
        });
        
        const shapes = [];
        for (let i = 0; i < 5; i++) {
            const shape = new THREE.Mesh(geometry, material.clone());
            shape.position.x = (Math.random() - 0.5) * 100;
            shape.position.y = (Math.random() - 0.5) * 100;
            shape.position.z = (Math.random() - 0.5) * 100;
            shape.scale.setScalar(3 + Math.random() * 5);
            shapes.push(shape);
            scene.add(shape);
        }
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            
            shapes.forEach((shape, idx) => {
                shape.rotation.x += 0.001 * (idx + 1);
                shape.rotation.y += 0.0015 * (idx + 1);
                shape.position.y += Math.sin(Date.now() * 0.001 + idx) * 0.01;
            });
            
            renderer.render(scene, camera);
        }
        
        animate();
        
        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    // --- Event Listener for Single Action ---
    getActionBtn.addEventListener('click', async () => {
        // Add click animation
        getActionBtn.classList.add('animate__animated', 'animate__pulse');
        setTimeout(() => {
            getActionBtn.classList.remove('animate__animated', 'animate__pulse');
        }, 1000);
        
        const batteryCharge = parseFloat(batteryChargeInput.value);
        const netEnergy = parseFloat(netEnergyInput.value);
        const avgFutureNetEnergy = parseFloat(avgFutureNetEnergyInput.value);
        const timeStep = parseInt(timeStepInput.value);

        if (isNaN(batteryCharge) || isNaN(netEnergy) || isNaN(avgFutureNetEnergy) || isNaN(timeStep)) {
            showErrorAlert('Please enter valid numerical values for all fields.');
            return;
        }
        if (timeStep < 0 || timeStep > 47) {
            showErrorAlert('Time Step must be between 0 and 47.');
            return;
        }

        const data = {
            battery_charge: batteryCharge,
            net_energy: netEnergy,
            avg_future_net_energy: avgFutureNetEnergy,
            time_step: timeStep
        };

        try {
            actionDisplay.textContent = 'Fetching...';
            actionDisplay.style.color = 'var(--gray-dark)';
            descriptionDisplay.textContent = '';
            qValuesDisplay.textContent = '';

            const response = await fetch('http://127.0.0.1:8000/api/predict_action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();

            // Animate the result display
            actionDisplay.textContent = result.action;
            actionDisplay.style.color = 'var(--success)';
            actionDisplay.classList.add('animate__animated', 'animate__fadeIn');
            
            setTimeout(() => {
                descriptionDisplay.textContent = result.action_description;
                descriptionDisplay.classList.add('animate__animated', 'animate__fadeIn');
                
                setTimeout(() => {
                    qValuesDisplay.textContent = JSON.stringify(result.predicted_q_values, null, 2);
                    qValuesDisplay.classList.add('animate__animated', 'animate__fadeIn');
                }, 200);
            }, 200);

        } catch (error) {
            console.error('Error fetching action:', error);
            showErrorAlert(`An error occurred: ${error.message}`);
            actionDisplay.textContent = `Error: ${error.message}`;
            actionDisplay.style.color = 'var(--danger)';
            descriptionDisplay.textContent = '';
            qValuesDisplay.textContent = '';
        }
    });

    // --- Charting Functions ---
    function createOrUpdateChart(chartInstance, ctx, type, labels, datasets, title) {
        if (chartInstance) {
            chartInstance.data.labels = labels;
            chartInstance.data.datasets = datasets;
            chartInstance.update();
            return chartInstance;
        }

        return new Chart(ctx, {
            type: type,
            data: {
                labels: labels,
                datasets: datasets,
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    },
                },
                scales: {
                    x: {
                        grid: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: 'Time Step (Hours)',
                            color: 'var(--gray-dark)'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        title: {
                            display: true,
                            text: 'Value (kWh)',
                            color: 'var(--gray-dark)'
                        },
                        beginAtZero: true
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuart'
                }
            }
        });
    }

    function renderCharts(simulationData) {
        const timeSteps = simulationData.map(d => d.time_step);

        // Battery Charge Chart
        const batteryCtx = document.getElementById('batteryChart').getContext('2d');
        const batteryData = simulationData.map(d => d.battery_charge);
        batteryChartInstance = createOrUpdateChart(
            batteryChartInstance,
            batteryCtx,
            'line',
            timeSteps,
            [{
                label: 'Battery Charge (kWh)',
                data: batteryData,
                borderColor: 'var(--primary)',
                backgroundColor: 'rgba(67, 97, 238, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }],
            'Battery Charge Over Time'
        );

        // Energy Flow Chart
        const energyFlowCtx = document.getElementById('energyFlowChart').getContext('2d');
        energyFlowChartInstance = createOrUpdateChart(
            energyFlowChartInstance,
            energyFlowCtx,
            'line',
            timeSteps,
            [
                { 
                    label: 'Solar Generation', 
                    data: simulationData.map(d => d.solar_generation), 
                    borderColor: 'var(--warning)',
                    backgroundColor: 'rgba(248, 150, 30, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                { 
                    label: 'Wind Generation', 
                    data: simulationData.map(d => d.wind_generation), 
                    borderColor: 'var(--success)',
                    backgroundColor: 'rgba(76, 201, 240, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                { 
                    label: 'Total Demand', 
                    data: simulationData.map(d => d.total_demand), 
                    borderColor: 'var(--danger)',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    tension: 0
                },
                { 
                    label: 'Grid Import', 
                    data: simulationData.map(d => d.grid_import), 
                    borderColor: 'var(--secondary)',
                    borderWidth: 2,
                    tension: 0.4
                },
                { 
                    label: 'Grid Export', 
                    data: simulationData.map(d => d.grid_export), 
                    borderColor: 'var(--accent)',
                    borderWidth: 2,
                    tension: 0.4
                }
            ],
            'Energy Flow Over Time'
        );

        // Loss Chart
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        lossChartInstance = createOrUpdateChart(
            lossChartInstance,
            lossCtx,
            'bar',
            timeSteps,
            [
                {
                    label: 'Unmet Demand',
                    data: simulationData.map(d => d.unmet_demand),
                    backgroundColor: 'rgba(239, 35, 60, 0.7)',
                    borderColor: 'var(--danger)',
                    borderWidth: 1
                },
                {
                    label: 'Wasted Energy',
                    data: simulationData.map(d => d.wasted_energy),
                    backgroundColor: 'rgba(76, 201, 240, 0.7)',
                    borderColor: 'var(--success)',
                    borderWidth: 1
                }
            ],
            'Unmet Demand & Wasted Energy Over Time'
        );
    }

    // --- Event Listener for Full Simulation ---
    runSimulationBtn.addEventListener('click', async () => {
        // Add loading animation
        runSimulationBtn.disabled = true;
        simulationStatus.textContent = 'Running simulation... Please wait.';
        simulationStatus.style.color = 'var(--gray-dark)';
        
        // Add loading dots animation
        let dots = 0;
        const dotInterval = setInterval(() => {
            dots = (dots + 1) % 4;
            simulationStatus.textContent = 'Running simulation' + '.'.repeat(dots) + ' '.repeat(3 - dots);
        }, 500);

        try {
            const response = await fetch('http://127.0.0.1:8000/api/simulate_episode', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
            }

            const simulationResult = await response.json();
            
            if (simulationResult.length > 0) {
                // Animate the charts appearing
                document.querySelectorAll('.chart-container').forEach((container, index) => {
                    container.style.opacity = '0';
                    container.style.transform = 'translateY(20px)';
                    container.style.transition = `all 0.5s ease ${index * 0.2}s`;
                    
                    setTimeout(() => {
                        container.style.opacity = '1';
                        container.style.transform = 'translateY(0)';
                    }, 100);
                });
                
                renderCharts(simulationResult);
                simulationStatus.textContent = `Simulation complete. Displaying data for ${simulationResult.length} time steps.`;
                simulationStatus.style.color = 'var(--success)';
            } else {
                simulationStatus.textContent = 'Simulation completed, but no data was returned.';
                simulationStatus.style.color = 'var(--warning)';
            }

        } catch (error) {
            console.error('Error running simulation:', error);
            simulationStatus.textContent = `Error: ${error.message}`;
            simulationStatus.style.color = 'var(--danger)';
            showErrorAlert(`An error occurred during simulation: ${error.message}`);
        } finally {
            clearInterval(dotInterval);
            runSimulationBtn.disabled = false;
        }
    });

    // Helper function to show error alerts
    function showErrorAlert(message) {
        const alert = document.createElement('div');
        alert.className = 'error-alert animate__animated animate__fadeInDown';
        alert.textContent = message;
        alert.style.position = 'fixed';
        alert.style.top = '20px';
        alert.style.right = '20px';
        alert.style.padding = '12px 20px';
        alert.style.background = 'var(--danger)';
        alert.style.color = 'white';
        alert.style.borderRadius = 'var(--border-radius-sm)';
        alert.style.boxShadow = 'var(--shadow-md)';
        alert.style.zIndex = '1000';
        
        document.body.appendChild(alert);
        
        setTimeout(() => {
            alert.classList.add('animate__fadeOutUp');
            setTimeout(() => {
                alert.remove();
            }, 500);
        }, 3000);
    }
});