# Задача теплопроводности методом продольно-поперечной прогонки с расспараллеливанием с помощью библиотеки MPI

## Постановка задачи

1. Размер сетки N = 30
2. Граничные условия
    - x_min = 0 
    - x_max = 1
    - y_min = 0
    - y_max = 0.5
3. Размер шага по пространству (шаг по оси X и Y)
    - dx = 0.03
    - dy = 0.016
4. Размер шага по времени
    - dt = 0.2
5. Граничные условия температуры
    - Tx_end = 1200
    - Tx_start = 600

Задача решается в двумерной постановке. С помощью метода продольно-поперечной прогонки проходимся по матрице `temperature[30 * 30]`. Суть заключается в том, что матрица сначала подается строками (по оси X) в метод `tridiagonal_matrix`, после вычислений матрица подается столбцами (по оси Y) в этот же метод. Этот метод также называется [ADI's method](https://en.wikipedia.org/wiki/Alternating-direction_implicit_method).

Прогонка матрицы по строкам и столбцам называется прогонка по слоям n + 1/2 и n + 1, где n - время нагревания.

1. Граничные условия по осям:
    - X : Tx_start -> Tx_end
    - Y : 600 * (1 + x) -> 600 * (1 + x * x * x), где x = x_min + rank * dx (rank - номер процесса от 0 до 30)

2. В методе `tridiagonal_matrix` вычисляются значения матрицы F и уже потом вычисляются значения матрицы температуры.

3. Распараллелвание происходит на 30 процессов. Размерность сетки равна количеству вычисляемых процессов, то есть каждая(ый) строка/столбец уходит на один процесс. Для реализация этого понадобилось создать производный тип MPI - `not_resized_columnType`. Также необходимо скорректировать его размер (4 байта) с помощью MPI_Type_create_resized - `columnType`.

4. Каждую строку lambdaX назначаем соответствующему процессу, каждый столбец lambdaY назначаем соответствущему процессу.

5. Инициализируем матрицу температуры с начальным значением 300 и матрицу F.

6. Начинаем первое прохождение по слою n + 1/2 (по оси X) на нулевом процессе (root).

7. В цикле начинаем прогонять матрицу уже со слоя n + 1 (по оси Y). С помощью MPI_Scatter распределяем столбцы матрицы температур и F по процессам через буферы отправки `temprerature` и `F` соответственно; совершаем вычисления над этими столбцами; собираем вычисленные столбцы матрицы температур и F в буфере приема `temperatureReceive` и `fReceive` соответственно. Также  переменная - `x`, которая передается в граничные условия оси Y, вычисляется в соответствии номеру процесса, на котором происходит вычисление.

8. Собираем вычисленные столбцы на нулевом процессе в буферах приема `temperature` и `F`. 

9. Проходим слой n + 1/2 по оси X. Распределяем уже строки по процессам, производим над ними вычислениями и передаем строки обратно на нулевой процесс. 

10. На процессах отличных от нулевого буфера отправки и приема в MPI_Scatter и MPI_Gather соответственно указывать опцианально.

11. Освобождаем память от производных типов.

12. То что в комментариях - последовательное решение. Передача в метод `tridiagonal_matrix` сначала построчной матрицы, затем матрица транспонируется и передача матрицы по столбцам.

[Статья аналогичная по сути. Очень помогла для решения, спасибо автору](https://habr.com/ru/articles/707462/)


[Видео-объяснение, которое полностью отражает концепцию задачи](https://www.youtube.com/watch?v=azAPv0i9K2c)


[Аналогичная задача сделанная в Excel](https://www.youtube.com/watch?v=JJaUw1cGrWU&t=576s)

