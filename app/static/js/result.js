const socket = io();

document.addEventListener('DOMContentLoaded', () => {
    socket.emit("get-result")

    document.getElementById("return-btn").onclick = () => {
        window.location.href = "/";
    }


    document.getElementById("calculate-btn").onclick = () => {
        const deltas = []
        document.querySelectorAll("#post_optimization input").forEach( function (input) {
            deltas.push(input.value ? input.value : 0)
        });
        socket.emit("post-optimization", {"deltas": deltas.reverse()})
    }
});

function generateInput(name, value, w_full='w-full', disabled=true) {
    const label = `
        <label 
            for="var-${name}" 
            class="block mb-2 text-sm font-medium text-white w-full max-w-72"
        >
            ${name}
        </label>
    `;

    const input = `
        <input
            type="number"
            id="var-${name}"
            value="${value}"
            class="block w-full p-2 max-w-72 border rounded-lg text-base border-gray-600 placeholder-gray-400 focus:ring-blue-500 focus:border-blue-500"
            ${disabled ? 'disabled' : ''}
        >
    `;

    return `
        <div class="flex flex-col relative items-center ${w_full}">
            ${label}
            ${input}
        </div>
    `;
}

socket.on('result', (data) => {
    const result = document.getElementById('result');
    data.x.forEach((value, key) => {
        result.innerHTML += generateInput(`x${key +1}`, value);
    })

    result.innerHTML += generateInput("Z", data.fun);

    const shadow_price = document.getElementById('shadow_price');
    data.shadow_prices.forEach((value, key) => {
        shadow_price.innerHTML += generateInput(`sx${key +1}`, value);
    })

    const tableau = document.getElementById('tableau');
    data.tableau.forEach((row, rowIndex) => {
        const tr = document.createElement('tr');
        row.forEach((cell, colIndex) => {
            const cellElement = rowIndex === 0 ? 'th' : 'td';
            const td = document.createElement(cellElement);
            td.textContent = cell;
            td.className = 'border border-gray-300 px-4 py-2';

            if (rowIndex === 0 && colIndex === 0) {
                td.classList.add('bg-red-200', 'font-bold');
            } else if (rowIndex === 0) {
                td.classList.add('bg-blue-200', 'font-bold');
            } else if (colIndex === 0) {
                td.classList.add('bg-green-200', 'font-bold');
            }

            tr.appendChild(td);
        });
        tableau.appendChild(tr);
    });

    const post_optimization = document.getElementById('post_optimization');
    data.x.forEach((value, key) => {
        post_optimization.innerHTML += generateInput(`Δ${key +1}`, 0, "", false);
    })
})

socket.on('post-optimization-result', (data) => {
    document.getElementById("z-new").value = data["z"];
    document.getElementById("message").innerHTML = data["message"];
})