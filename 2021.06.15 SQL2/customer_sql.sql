use classicmodels;
show tables;
select * from employees;
select firstName from employees;
select firstName, lastName from employees;
select employeeNumber, firstName, lastName from employees where employees.employeeNumber >= 1300;
select * from offices;
select city from offices where offices.officeCode='1';
select city, phone from offices where offices.officeCode='1';
select customers.customerNumber, orders.orderNumber, customers.country from orders 
LEFT JOIN customers ON customers.customerNumber = orders.customerNumber;
select customers.customerNumber, payments.checkNumber from customers 
LEFT JOIN payments on customers.customerNumber = payments.customerNumber;

SELECT customers.customerNumber, orders.orderNumber, customers.country from orders 
INNER JOIN customers ON orders.customerNumber = customers.customerNumber
WHERE customers.country ='USA';











select customers.state, customers.customerName, payments.checkNumber from customers 
LEFT JOIN payments on customers.customerNumber = payments.customerNumber
where payments.paymentDate >= '2003-06-06';