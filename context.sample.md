# Overview
Human resources analytics database for employee and department reporting.

# Dialect
PostgreSQL

# Tables
## employees
Description: Master employee records with salary and organization mapping.
Columns:
- id
- name
- department
- salary
- hire_date

## departments
Description: Department-level planning and budget allocations.
Columns:
- department
- budget
- region

# Relationships
- employees.department = departments.department

# Rules
- Return exactly one read-only SQL query.
- Use explicit joins instead of implicit joins.
- Prefer clear aliases for readability.
- Avoid SELECT * unless the user explicitly requests it.

# Examples
- Question: Top 5 departments by average salary where payroll stays within budget.
