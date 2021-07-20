# Contenido
* Introducción
* Rectas: Concepto de recta y pensar en cómo encontrar los parámetros
* Mínimos Cuadrados
* Ecuación Normal: Derivando la ecuación, convexidad del error cuadrático, mostrando el resultado del ajuste y los parámetros
* Normalización de los datos
* Múltiples regresiones lineales: Extender la idea al plano
* Interpretación de la regresión: Qué significan los parámetros
* Motivación para el iterativo: En la vida real se pueden tener datos masivos, producto vector matriz es factible
* Descenso del gradiente: Ejemplo resolviendo una ecuación no lineal, Definición para Reg Lineal
* Ajuste: Visualizando parámetros conforme ajusta

Para el sgte vid: Ajuste polinomial, prueba de hipótesis..., Lasso, Ridge, Otros métodos iterativos

### Introducción:
Cuando se tienen datos de la siguiente forma,
a veces es necesario predecir valores fuera del rango conocido

Por ejemplo, si decimos que $x$ representa el ingreso semanal de una persona

y $y$ el gasto de consumo semanal

Sería útil poder saber el precio estimado de una casa dado algún área

pero cómo podemos estimar el mejor valor en base a la información disponible?



La idea consiste en encontrar la línea de mejor ajuste a los puntos,

y cómo lo conseguimos habiendo tantas rectas posibles?

### Rectas 1
Recordemos que una recta está definida por su pendiente $m$ e intercepto $b$
`ecuación de la recta`
donde la pendiente indica la inclinación de la recta como la proporción del cambio vertical con respecto al horizontal,  y el intercepto en qué punto toca el eje $y$



En la literatura estadística se les llama $\beta_1$ al intercepto y $\beta_2$ a la pendiente.
variando estos dos valores podemos construir infinitas rectas.
`Mostrando rectas con parámetros aleatorios`

### Rectas 2

Ya conocemos la entrada y respuesta de algunos puntos

modelamos esta información asumiento que tienen una relación lineal y pueden ser explicados por una recta `line_eq1`, a esto se le agrega un término de error, el cual representa variables no consideradas o ruido `line_eq2` en nuestro ejemplo el gasto mensual no solamente depende del ingreso, existen variables del diario vivir que no se están considerando

Ahora queremos predecir el consumo para valores desconocidos pero que provienen de la misma distribución de datos.

Para estas observaciones nuevas no podemos calcular el término de error, por lo que lo omitimos, lo cual hará que nuestra estimación no sea perfecta, pero si lo más cercana al valor real posible

A esta estimación la llamaremos $\hat{y}$.

Así podemos definir una forma de medir, dados $\beta_1$ y $\beta_2$ cuán distinta es la estimación del valor $\hat{y}$ al valor original $y_i$ para los puntos $(x_i, y_i)$ conocidos.

Promediando estos valores para todos los puntos obtenemos una medida de error total.

Esta medida de error se llama error cuadrático medio: $\displaystyle E(\beta)=\frac{1}{m}\sum_{i=1}^m (\hat{y}_i-y_i)^2$, y es una de las funciones de error más utilizadas.

Ahora sabemos que una línea ajusta mejor mientras menor el valor de $E(\beta)$

por lo que necesitamos una manera automática de dado un conjunto de datos $X$ y $Y$, obtener los valores de $\beta$ que produzcan el menor valor posible de $J(\beta)$

### Ecuación Normal 1 (Notación)
Para simplificar los cálculos es conveniente realizarlos en notación matricial.

Como tenemos $m$ puntos tambien llamados observaciones en nuestro conjunto de datos

Podemos reordenar la respuestas conocidos como un vector de resultados $\mathbf{\hat{y}}$ donde cada elemento o "fila" corresponde a una observación.

$$\mathbf{y}=\begin{bmatrix}\beta_1+\beta_2x_1\\\beta_1+\beta_2x_2\\\vdots\\\beta_1+\beta_2x_m\end{bmatrix}$$

Ahora podemos reescribir la suma de esta manera $\beta_1\cdot1 + \beta_2\cdot x$ escribiendo de manera explícita el uno por $\beta_1$

`Mover a la izquierda y empezar a escribir a la derecha` recordando la definición del producto punto entre dos vectores.

Si tenemos $\mathbf{x}_i = \begin{bmatrix}1\\ x_i\end{bmatrix}$ y $\beta = \begin{bmatrix}\beta_1\\ \beta_2\end{bmatrix}$, su producto punto es $\langle \mathbf{x}_i, \beta\rangle = 1\cdot \beta_1 + x_i\cdot\beta_2$.

`De nuevo a la izquierda` entonces cada elemento del vector es el producto punto entre los vectores 

$$\mathbf{y} = \begin{bmatrix}\langle\begin{bmatrix}1\\x_1\end{bmatrix},\begin{bmatrix}\beta_1\\\beta_2\end{bmatrix}\rangle\\ \langle\begin{bmatrix}1\\x_2\end{bmatrix},\begin{bmatrix}\beta_1\\\beta_2\end{bmatrix}\rangle\\ \vdots  \\ \langle\begin{bmatrix}1\\x_m\end{bmatrix},\begin{bmatrix}\beta_1\\\beta_2\end{bmatrix}\rangle\end{bmatrix}$$

`A la derecha` Pero podemos reescribir el producto punto como el vector $\mathbf{x}_i$ traspuesto volviendose un vector fila, por el vector $\beta$ así. $\mathbf{x}_i^T\beta = \begin{bmatrix}1, x_i\end{bmatrix}\begin{bmatrix}\beta_1\\ \beta_2\end{bmatrix}$ 

Esta notación se basa en considerarlos directamente como matrices de $v^T: 1\times2$ y $u: 2\times 1$ respectivamente.

`Izquierda` Reescribiendo así el vector $\mathbf{y}$:

$$\mathbf{y} = \begin{bmatrix}\mathbf{x}_1^T\beta\\ \mathbf{x}_2^T\beta\\ \vdots  \\ \mathbf{x}_m^T\beta\end{bmatrix}$$

Hasta este punto ya simplificamos mucho la notación, pero notemos que $\beta$ se repite muchas veces, podemos simplificarla más considerando cada vector $\mathbf{x}_i^T$ como fila de una matriz.

`Borrar la derecha` De modo que tenemos la matriz $X = \begin{bmatrix}1, x_1\\ 1, x_2\\\vdots\\1, x_m\end{bmatrix}$ llamada matriz de diseño. Al multiplicarla por el vector $\beta$ obtenemos `multiplicación larga` que se escribe matricialmente como $\mathbf{\hat{y}} = \mathbf{X}\beta$.

Esta notación es muy util para simplificar los cálculos y escribir programas que lo resuelvan.

### Mínimos Cuadrados

Recordemos que todo sistema de ecuaciones

$$\begin{cases}a_{11}x_1+a_{12}x_2+\dots+a_{1n}x_n=b_1\\ a_{21}x_1+a_{22}x_2+\dots+a_{2n}x_n=b_2\\\vdots\\a_{n1}x_1+a_{n2}x_2+\dots+a_{nn}x_n=b_n\end{cases}$$

se puede reescribir como una ecuación matricial.

$$\begin{bmatrix}a_{11}, a_{12}, \dots, a_{1n}\\a_{21}, a_{22}, \dots, a_{2n}\\\vdots\\a_{n1}, a_{n2}, \dots, a_{nn}\end{bmatrix}\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}=\begin{bmatrix}b_1\\b_2\\\vdots\\b_n\end{bmatrix}$$

$$A\mathbf{x}=\mathbf{b}$$

Existen casos donde el sistema no tiene solución, sin embargo existirá un vector $\mathbf{x}^*$ que al multiplicarse por $A$ da como resultado un vector $b^*$ lo más cercano posible a $b$.

Al vector $\mathbf{x}^*$ se le llama solución por mínimos cuadrados.

Esta formulación es equivalente a encontrar el $\mathbf{x}^*$ que minimice ($\underset{\mathbf{x}}{\operatorname{argmin}} ||A\mathbf{x}-b||^2$)

### Ecuación Normal 2

Ahora se puede reescribir el error cuadrático medio como `ms1 a mse2` de manera matricial, para enccontrar el mínimo de esta función podemos prescindir el término $\frac{1}{m}$ simplificandose a: `mse3`

--------------------------------------

Recordemos que $\mathbf{\hat{y}} = X\beta$

**Esta formulación es una reescritura de la norma descrita en mínimos cuadrados (Escribir la ecuación de mínimos cuadrados con $\beta$** 

------------------------------------------------------

`Mover arriba y empezar el desarrollo` Desarrollando esta expresión tenemos.

La función del error cuadrático medio forma una superficie convexa, es decir que tiene un único punto mínimo. `Graficar el MSE`
Por esta razón se puede derivar la expresión con respecto a $\beta$

E igualarla a 0, técnica común para encontrar mínimos.

`Desarrollar`
Obtenemos la siguiente expresión, que nos da una fórmula exacta para los β que producen el menor error. `Enfatizar la fórmula`

### Ecuación Normal 3

`Graficando` Regresando al conjunto de puntos del inicio

`Hacer pequeña la gráfica y mandarla a la izquierda` 

`mostrar el matrices con pocos datos`

Si se resuelve la ecuación anterior `mostrar ecuación` 

Se obtiene el vector de parámetros

Graficando la recta obtenida por estos parámetros podemos observar el ajuste obtenido.

### Multiples Regresiones Lineales



### Interpretación de la regresión

### Normalización de datos

Como las variables en la vida real tienen distintas escalas, esto puede hacer dificil de interpretar los parámetros, para evitar esto, y simplificar la función de error se estandarizan los datos.

Vamos a denotar la segunda colúmna de la matriz X como $X_{:, 2}$

Para la segunda colúmna de $X$ se calcula la media y su desviación estándar, 

y se realiza el cálculo elemento a elemento en la colúmna, que se guarda en la segunda colúmna de la matriz estandarizada X tilde. Esta matriz también tiene unos en su primera colúmna.

Se realiza el mismo procedimiento para $\mathbf{y}$

`Graficar puntos en escala real y llevarlos a la normal` aplicandolo, observamos como se desplazan los puntos al centro

`Zoom al grafico`

`Graficar Superficie de error` la superficie de error es el siguiente paraboloide:

El punto $(\beta_1, \beta_2, E(\beta))$ está lo más cerca posible al mínimo de la superficie de error como se puede observar `Graficar el punto mínimo`.

Con los valores de los betas y el error calculados `Evaluar el punto mínimo`

Tenemos la línea de mejor ajuste en los datos estandarizados

Nótese que al estandarizar, el intercepto se vuelve cero.

### Motivación

[Revisar](https://stats.stackexchange.com/questions/278755/why-use-gradient-descent-for-linear-regression-when-a-closed-form-math-solution)

En la vida real, la matriz $X$ puede llegar a ser gigante con una gran cantidad de datos.

Veamos una vez más la fórmula de la ecuación normal $\beta = (\mathbf{X}^T\cdot\mathbf{X})^{-1}\cdot\mathbf{X}^T\cdot\mathbf{y}$, notemos que tenemos que multiplicar $X^TX$ si suponemos que $X$ tiene dimensiones $m\times n$, una estimación de la cantidad de operaciones para calcularlo es $O(m^2n)$ [notación O indica el peor caso](https://www.freecodecamp.org/news/big-o-notation-simply-explained-with-illustrations-and-video-87d5a71c0174/) 

Una computadora moderna puede realizar unas $10^8$ operaciones por segundo en un lenguaje veloz como **c++** y **Fortran** que es en lo que están escritas las librerías de algebra lineal computacional

Si tuvieramos una matriz de $300000\times 1000$, realizar la operación tomaría unas $10^{13}$ operaciones

Es plausible esperar unos segundos para esto, sin embargo con $300000\times 10000$ un caso que se puede dar en información geoespacial, ya no cabe en la memoria de una computadora común, es por estas limitaciones de tiempo y memoria que se plantea encontrar los parámetros de manera iterativa.

