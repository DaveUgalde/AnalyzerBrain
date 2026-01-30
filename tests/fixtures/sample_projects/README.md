Pruebas

bash
pytest tests/
''')

class JavaScriptProject(SampleProject):
"""Proyecto JavaScript/Node.js simple."""

text
def __init__(self):
    super().__init__("javascript_simple")

def _create_structure(self):
    # package.json
    package_json = os.path.join(self.path, "package.json")
    with open(package_json, 'w') as f:
        f.write('''{
"name": "sample-js-project",
"version": "1.0.0",
"description": "JavaScript sample project for testing",
"main": "index.js",
"scripts": {
"start": "node index.js",
"test": "jest",
"lint": "eslint ."
},
"dependencies": {
"express": "^4.18.0"
},
"devDependencies": {
"jest": "^29.0.0",
"eslint": "^8.0.0"
}
}
''')

text
    # index.js
    index_js = os.path.join(self.path, "index.js")
    with open(index_js, 'w') as f:
        f.write('''/**
Aplicación JavaScript de ejemplo.
*/
const express = require('express');

class Calculator {
constructor() {
this.history = [];
}

text
add(a, b) {
    const result = a + b;
    this.recordOperation('add', [a, b], result);
    return result;
}

subtract(a, b) {
    const result = a - b;
    this.recordOperation('subtract', [a, b], result);
    return result;
}

multiply(a, b) {
    const result = a * b;
    this.recordOperation('multiply', [a, b], result);
    return result;
}

divide(a, b) {
    if (b === 0) {
        throw new Error('Division by zero');
    }
    const result = a / b;
    this.recordOperation('divide', [a, b], result);
    return result;
}

recordOperation(operation, operands, result) {
    this.history.push({
        operation,
        operands,
        result,
        timestamp: new Date().toISOString()
    });
}

getHistory() {
    return this.history;
}

clearHistory() {
    this.history = [];
}
}

// Utilidades
function formatCurrency(amount, currency = 'USD') {
return new Intl.NumberFormat('en-US', {
style: 'currency',
currency: currency
}).format(amount);
}

function validateEmail(email) {
const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
return pattern.test(email);
}

// Servidor Express simple
function createServer() {
const app = express();
const calculator = new Calculator();

text
app.use(express.json());

app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.post('/api/calculate', (req, res) => {
    try {
        const { operation, a, b } = req.body;
        let result;
        
        switch (operation) {
            case 'add':
                result = calculator.add(a, b);
                break;
            case 'subtract':
                result = calculator.subtract(a, b);
                break;
            case 'multiply':
                result = calculator.multiply(a, b);
                break;
            case 'divide':
                result = calculator.divide(a, b);
                break;
            default:
                throw new Error('Invalid operation');
        }
        
        res.json({
            success: true,
            result,
            historyLength: calculator.getHistory().length
        });
    } catch (error) {
        res.status(400).json({
            success: false,
            error: error.message
        });
    }
});

app.get('/api/history', (req, res) => {
    res.json(calculator.getHistory());
});

return app;
}

// Exportar para pruebas
if (require.main === module) {
const app = createServer();
const port = process.env.PORT || 3000;

text
app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
}

module.exports = {
Calculator,
formatCurrency,
validateEmail,
createServer
};
''')

text
    # test/index.test.js
    test_dir = os.path.join(self.path, "test")
    os.makedirs(test_dir, exist_ok=True)
    
    test_js = os.path.join(test_dir, "index.test.js")
    with open(test_js, 'w') as f:
        f.write('''const { Calculator, formatCurrency, validateEmail } = require('../index.js');
describe('Calculator', () => {
let calculator;

text
beforeEach(() => {
    calculator = new Calculator();
});

test('adds numbers correctly', () => {
    expect(calculator.add(2, 3)).toBe(5);
    expect(calculator.add(-1, 1)).toBe(0);
});

test('subtracts numbers correctly', () => {
    expect(calculator.subtract(5, 3)).toBe(2);
    expect(calculator.subtract(0, 5)).toBe(-5);
});

test('multiplies numbers correctly', () => {
    expect(calculator.multiply(2, 3)).toBe(6);
    expect(calculator.multiply(0, 5)).toBe(0);
});

test('divides numbers correctly', () => {
    expect(calculator.divide(6, 3)).toBe(2);
    expect(calculator.divide(5, 2)).toBe(2.5);
});

test('throws error when dividing by zero', () => {
    expect(() => calculator.divide(5, 0)).toThrow('Division by zero');
});

test('records history', () => {
    calculator.add(1, 2);
    calculator.multiply(3, 4);
    
    const history = calculator.getHistory();
    expect(history).toHaveLength(2);
    expect(history[0].operation).toBe('add');
    expect(history[1].operation).toBe('multiply');
});
});

describe('Utilities', () => {
test('formats currency correctly', () => {
expect(formatCurrency(1234.56)).toBe('$1,234.56');
expect(formatCurrency(0.99)).toBe('$0.99');
});

text
test('validates email addresses', () => {
    expect(validateEmail('test@example.com')).toBe(true);
    expect(validateEmail('invalid-email')).toBe(false);
    expect(validateEmail('@example.com')).toBe(false);
});
});
''')

class MultiLanguageProject(SampleProject):
"""Proyecto con múltiples lenguajes."""

text
def __init__(self):
    super().__init__("multi_language")

def _create_structure(self):
    # Python
    python_dir = os.path.join(self.path, "python")
    os.makedirs(python_dir, exist_ok=True)
    
    py_file = os.path.join(python_dir, "utils.py")
    with open(py_file, 'w') as f:
        f.write('''
def python_function():
"""Función Python."""
return "Hello from Python"
''')

text
    # JavaScript
    js_dir = os.path.join(self.path, "javascript")
    os.makedirs(js_dir, exist_ok=True)
    
    js_file = os.path.join(js_dir, "utils.js")
    with open(js_file, 'w') as f:
        f.write('''
function jsFunction() {
// Función JavaScript
return "Hello from JavaScript";
}

module.exports = { jsFunction };
''')

text
    # Java
    java_dir = os.path.join(self.path, "java", "src", "main", "java", "com", "example")
    os.makedirs(java_dir, exist_ok=True)
    
    java_file = os.path.join(java_dir, "Main.java")
    with open(java_file, 'w') as f:
        f.write('''
package com.example;

public class Main {
public static void main(String[] args) {
System.out.println("Hello from Java");
}

text
public static String javaMethod() {
    return "Java method";
}
}
''')

text
    # README
    readme = os.path.join(self.path, "README.md")
    with open(readme, 'w') as f:
        f.write('''# Proyecto Multi-Lenguaje
Proyecto con código en múltiples lenguajes para pruebas.

Lenguajes incluidos

Python
JavaScript
Java
Propósito

Probar análisis multi-lenguaje de Project Brain.
''')

def create_sample_project(project_type="python_simple"):
"""
Factory para crear proyectos de muestra.

text
Args:
    project_type: Tipo de proyecto a crear
    
Returns:
    SampleProject instance
"""
projects = {
    "python_simple": PythonSimpleProject,
    "javascript": JavaScriptProject,
    "multi_language": MultiLanguageProject
}

if project_type not in projects:
    raise ValueError(f"Tipo de proyecto no soportado: {project_type}")

return projects[project_type]()
Proyectos pre-creados para fácil acceso

SAMPLE_PROJECTS = {
"python": PythonSimpleProject(),
"javascript": JavaScriptProject(),
"multi_lang": MultiLanguageProject()
}

text

Este sistema de pruebas completo implementa:

1. **Configuración pytest** (`conftest.py`) con:
   - Fixtures para todos los componentes
   - Configuración de entorno de prueba
   - Manejo de pruebas asíncronas
   - Helpers de utilidad

2. **Pruebas unitarias** para:
   - Core/Orchestrator
   - Indexer/Parser
   - Embeddings/Generator
   - Agents/Base

3. **Pruebas de integración** para:
   - Componentes core trabajando juntos
   - Flujos básicos del sistema

4. **Pruebas end-to-end** para:
   - Flujo completo de análisis
   - Proyectos reales
   - Recuperación de errores

5. **Pruebas de performance** para:
   - Parsing de archivos grandes
   - Generación de embeddings
   - Operaciones concurrentes
   - Benchmarks comparativos

6. **Fixtures** con:
   - Proyectos de muestra en diferentes lenguajes
   - Datos de prueba estructurados
   - Configuraciones predefinidas
