async function predictPrice() {
    // 1. Collect Data from the HTML Form
    // Note: We use the IDs from the new 'Enterprise' HTML layout
    const data = {
        brand: document.getElementById('brand').value,
        model: document.getElementById('model').value,
        year: document.getElementById('year').value,
        km_driven: document.getElementById('km_driven').value,
        mileage: document.getElementById('mileage').value,
        engine: document.getElementById('engine').value,
        max_power: document.getElementById('max_power').value,
        seats: document.getElementById('seats').value,
        fuel: document.getElementById('fuel').value,
        transmission: document.getElementById('transmission').value,
        seller_type: document.getElementById('seller_type').value,
        owner: document.getElementById('owner').value
    };

    // Simple validation to prevent sending empty data
    if (!data.brand || !data.model) {
        alert("Please select a Brand and Model first.");
        return;
    }

    // 2. Send Data to Python Backend via Fetch API
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // 3. Update the UI with the result
        if (result.price) {
            const priceText = document.getElementById('priceText');
            const resultCard = document.getElementById('result');

            // Set the price
            priceText.innerText = "₹ " + result.price;

            // Make the result card visible (removing 'hidden' class and setting visibility)
            resultCard.classList.remove('hidden');
            resultCard.style.visibility = 'visible';
            
            // Smooth scroll to the result (great for mobile users)
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            
        } else {
            alert("Error from server: " + result.error);
        }

    } catch (error) {
        console.error("Error:", error);
        alert("Something went wrong! Please check the console.");
    }
}