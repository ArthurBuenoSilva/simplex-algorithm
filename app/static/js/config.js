document.addEventListener('DOMContentLoaded', function () {
    const varQtdInput = document.getElementById('var-qtd');
    const restrQtdInput = document.getElementById('restrictions-qtd');
    const continueBtn = document.getElementById('continue-btn');

    function checkInputs() {
        const varQtd = parseInt(varQtdInput.value);
        const restrQtd = parseInt(restrQtdInput.value);
        if (varQtd > 0 && restrQtd > 0) {
            continueBtn.disabled = false;
            if (continueBtn.classList.contains("cursor-not-allowed")) {
                continueBtn.classList.remove("cursor-not-allowed");
            }
        } else {
            continueBtn.disabled = true;
            if (!continueBtn.classList.contains("cursor-not-allowed")) {
                continueBtn.classList.add("cursor-not-allowed");
            }
        }
    }

    varQtdInput.addEventListener('input', checkInputs);
    restrQtdInput.addEventListener('input', checkInputs);

    continueBtn.addEventListener('click', function () {
        const varQtd = parseInt(varQtdInput.value);
        const restrQtd = parseInt(restrQtdInput.value);
        const pplConfig = document.getElementById('ppl-config');
        pplConfig.innerHTML = '';

        let objectiveFunctionHtml = '<h3 class="text-white text-lg font-bold">Função objetivo</h3><form id="objective-function" class="flex flex-wrap space-x-4 w-full text-white items-center">';
        for (let i = 1; i <= varQtd; i++) {
            objectiveFunctionHtml += `<input type="number" class="text-black w-12 max-w-64 flex-1 p-2 mb-2 border rounded-lg text-base border-gray-600 placeholder-gray-400 focus:ring-blue-500 focus:border-blue-500" placeholder="x${i}">`;
            if (i < varQtd) {
                objectiveFunctionHtml += '<span class="h-full place-content-center mb-2.5">+</span>';
            }
        }
        objectiveFunctionHtml += '<span class="h-full place-content-center mb-2.5">=</span><span class="h-full place-content-center mb-2.5">Z</span>';
        objectiveFunctionHtml += '</form>';

        let restrictionsHtml = '<h3 class="text-white text-lg font-bold">Restrições</h3><form id="restrictions" class="flex flex-col h-full w-full text-white overflow-y-auto space-y-4">';
        for (let i = 1; i <= restrQtd; i++) {
            restrictionsHtml += '<div class="flex flex-wrap space-x-4">';
            for (let j = 1; j <= varQtd; j++) {
                restrictionsHtml += `<input type="number" class="text-black w-12 max-w-64 flex-1 p-2 mb-2 border rounded-lg text-base border-gray-600 placeholder-gray-400 focus:ring-blue-500 focus:border-blue-500" placeholder="x${j}">`;
                if (j < varQtd) {
                    restrictionsHtml += '<span class="h-5/6 place-content-center mb-2.5">+</span>';
                }
            }
            restrictionsHtml += '<select class="text-black w-8 min-w-20 max-w-20 flex-1 p-2 mb-2 border rounded-lg text-base border-gray-600 placeholder-gray-400 focus:ring-blue-500 focus:border-blue-500"><option value="<=">&lt;=</option><option value=">=">&gt;=</option><option value=">">&gt;</option><option value="<">&lt;</option><option value="=">=</option></select> ';
            restrictionsHtml += `<input type="number" class="text-black w-12 max-w-64 flex-1 p-2 mb-2 border rounded-lg text-base border-gray-600 placeholder-gray-400 focus:ring-blue-500 focus:border-blue-500">`;
            restrictionsHtml += '</div>';
        }
        restrictionsHtml += '</form>';

        pplConfig.innerHTML = objectiveFunctionHtml + restrictionsHtml;

        const generateResultBtn = document.createElement('button');
        generateResultBtn.id = 'generate-result';
        generateResultBtn.type = 'button';
        generateResultBtn.className = 'text-white w-1/4 max-w-64 bg-gradient-to-r from-green-400 via-green-500 to-green-600 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-green-300 dark:focus:ring-green-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center me-2 mb-2 mt-4';
        generateResultBtn.textContent = 'Gerar resultado';

        const resultDiv = document.createElement('div');
        resultDiv.className = 'flex justify-end';
        resultDiv.appendChild(generateResultBtn);
        pplConfig.appendChild(resultDiv);

        generateResultBtn.addEventListener('click', function () {
            const data = {
                varQtd: varQtdInput.value, restrQtd: restrQtdInput.value, objectiveFunction: {}, restrictions: []
            };

            document.querySelectorAll('#objective-function input').forEach(function (input, key) {
                if (key < varQtd) {
                    data.objectiveFunction[`x${key + 1}`] = input.value ? input.value : 0;
                }
            });

            data.objectiveFunction["operator"] = "=";
            data.objectiveFunction["result"] = "Z";

            document.querySelectorAll('form#restrictions').forEach(function (form) {
                const restrictions = []
                let restriction = {};
                let count = 1;

                form.querySelectorAll('input, select').forEach(function (input) {
                    if (count === varQtd + 1) {
                        restriction["operator"] = input.value;
                    } else if (count === varQtd + 2) {
                        restriction["result"] = input.value ? input.value : 0;
                        restrictions.push(restriction)
                        count = 0;
                        restriction = {};
                    } else {
                        restriction[`x${count}`] = input.value ? input.value : 0;
                    }
                    count++;
                });
                data.restrictions = restrictions;
            });

            fetch('/generate_result', {
                method: 'POST', headers: {
                    'Content-Type': 'application/json'
                }, body: JSON.stringify(data)
            })
                .then(response => {
                    if (response.ok) {
                        window.location.href = '/result';
                    }
                });
        });
    });
})