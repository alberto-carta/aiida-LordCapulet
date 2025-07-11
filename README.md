# Custom AiiDA Calculation Plugin Quickstart

## 1. Project Structure
```
your_plugin/
├── setup.py
├── lordcapulet_custom_pw/
│   ├── __init__.py
│   └── custom_pw.py
```

## 2. Installation

From the plugin root directory (where `setup.py` is):
```bash
pip install -e .
```

## 3. Check Installation

- The package should appear with:
  ```bash
  pip show aiida_custom_pw
  ```

- The plugin should be listed with:
  ```bash
  verdi plugin list aiida.calculations | grep custom_pw
  ```

- Import in Python or `verdi shell` should work:
  ```python
  from aiida_custom_pw.custom_pw import CustomPwCalculation
  ```

- Restart the daemon after installation:
  ```bash
  verdi daemon restart
  ```

## 4. Usage in Scripts

Use your plugin via:
```python
from aiida_custom_pw.custom_pw import CustomPwCalculation
builder = CustomPwCalculation.get_builder()
builder.code = aiida.orm.load_code('your_code@your_computer')
# ... set other builder inputs ...
```
