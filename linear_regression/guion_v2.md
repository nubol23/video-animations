# Regresión Lineal

**Contenido**

* Introducción
* Rectas: Concepto de recta y pensar en cómo encontrar los parámetros
* Mínimos Cuadrados
* Ecuación Normal: Derivando la ecuación, convexidad del error cuadrático, mostrando el resultado del ajuste y los parámetros
* Múltiples regresiones lineales: Extender la idea al plano

### Introducción:

Seguro que alguna vez viste un gráfico de puntos similar a este.

Supongamos que en el eje $x$ se indica el ingreso semanal de una persona, y en el $y$, el gasto de consumo en esa cantidad de tiempo.

En otras palabras cada punto representa una persona y nos dice cuánto gasta en base a su ganancia.

Digamos que quieres saber cuánto consume aproximadamente alguien que gane 280, ¿cómo podemos conocer ese valor?

Pues este tipo de problemas de predicción se lo plantearon Gauss y Legendre `Escribir los nombres` mientras trataban de determinar las órbitas de cuerpos celestes en base a observaciones pasadas.

Descubrieron que podían aproximarlas con buena precisión al encontrar una curva que se ajuste mejor a los puntos, al caso donde la curva es una recta se le llamó **Regresión Lineal**

La pregunta ahora es, ¿cómo encontrar esta recta de entre todas las posibles?

### Rectas 1
Empecemos por comprender la ecuación de una recta, la cual está definida por su pendiente $m$ e intercepto $b$.
`ecuación de la recta`
Si trazamos una línea, la pendiente nos dice su inclinación, esta se calcula como la proporción del cambio vertical, respecto al horizontal, que en este ejemplo es $\frac{1}{2}$, mientras que el intercepto, representa en qué punto toca el eje $y$

En la literatura estadística se les llama $\beta_1$ al intercepto y $\beta_2$ a la pendiente.

Es variando estos dos parámetros, que podemos construir cualquier recta en el espacio.
`Mostrando rectas con parámetros aleatorios`

### Rectas 2

Ya conocemos las coordenadas de algunos puntos, modelamos esta información con una función que recibe el i-esimo valor en $x$ y devuelve su correspondiente en $y$

Esta función se define asumiendo que tienen una relación lineal y pueden ser explicados por una recta `line_eq1` , a esto se le agrega un término de error `lineas azules`, el cual representa variables no consideradas o ruido `line_eq2`.

Por ejemplo el gasto mensual no solamente depende del ingreso, existen variables en el día a día de una persona que no se están considerando.

Para observaciones nuevas no podemos calcular el término de error, por lo que lo omitimos, lo cual hará que nuestra aproximación no sea perfecta, pero si lo más cercana al valor real 

A esta estimación la llamaremos $\hat{y} $, además asumiremos que nuestros puntos son una muestra, por lo que reescribimos los parámetros con el sombrero para indicar que son una estimación y se les llama estimadores.

Necesitamos una forma de medir, cuán distinta es la predicción del valor $\hat{y}$ al original para los puntos conocidos.

Esto se hace restando los valores, <u>para darle más peso a las diferencias grandes e igualdad de importancia a los errores positivos y negativos</u>, se eleva al cuadrado.

Promediando estos valores obtenemos una medida de error, llamada error cuadrático medio: $\displaystyle J(\beta)=\frac{1}{m}\sum_{i=1}^m (\hat{y}_i-y_i)^2$, y es una de las funciones de costo más utilizadas.

Una línea se ajusta mejor a los puntos mientras más pequeño el valor de J, por lo que necesitamos una manera automática de dado un conjunto de datos $X$ y $Y$, obtener los valores de los $\beta$ que produzcan el menor error.

Esta operación se llama argumento del mínimo.

### Disclaimer

A partir de aquí, se detalla cómo obtener la fórmula para calcular el vector de parámetros $\hat{\beta}$ mediante mínimos cuadrados.

Si ya la conoces, o sólo te interesa comprender gráficamente qué hace la regresión, avanza al minuto: ...

### Ecuación Normal 1 (Notación)

--------------------------------------------------------------------------------

Para resolver esto debemos notar primero que todo sistema de ecuaciones lineales

se puede expresar en función de matrices.

Como tenemos $m$ puntos en nuestro conjunto de datos

Podemos acomodar las estimaciones de las respuestas en un vector donde cada elemento corresponde a una observación.

$$\begin{bmatrix}\hat{y}_1\\\hat{y}_2\\\vdots\\\hat{y}_m\end{bmatrix}=\begin{bmatrix}\beta_1+\beta_2x_1\\\beta_1+\beta_2x_2\\\vdots\\\beta_1+\beta_2x_m\end{bmatrix}$$

Ahora podemos reescribir la operación anotando el uno acompañando al primer parámetro de manera explícita.

`Mover a la izquierda y empezar a escribir a la derecha` 

Recordando la definición del producto punto entre dos vectores.

Si tenemos un vector $\mathbf{x}_i = \begin{bmatrix}1\\ x_i\end{bmatrix}$ y otro $\beta = \begin{bmatrix}\beta_1\\ \beta_2\end{bmatrix}$ conteniendo los estimadores, su producto punto se defiune como  la siguiente expresión $\langle \mathbf{x}_i, \beta\rangle = 1\cdot \beta_1 + x_i\cdot\beta_2$.

`De nuevo a la izquierda`

Así cada elemento del vector es un producto de vectores 

`A la derecha` Podemos reescribirlo como $\mathbf{x}_i$ traspuesto volviendose un vector fila, por $\beta$.

`Izquierda` aplicamos esto al vector

y vemos que hasta este punto ya simplificamos mucho la notación, pero nota que $\beta$ se repite muchas veces, entonces consideremos cada vector $\mathbf{x}_i^T$ como fila de una matriz.

`Borrar la derecha` De modo que tenemos X mayúscula, esta estructura se llama llamada matriz de diseño. Al multiplicarla por el vector $\beta$ `multiplicación larga` se tiene una ecuación en forma matricial $\mathbf{\hat{y}} = \mathbf{X}\beta$.

### Mínimos Cuadrados

Como todo se reduce a resolver un sistema de ecuaciones lineales, es muy importante mencionar que hay casos para los que no existe solución, pero, se puede hallar un vector $\mathbf{x}^*$ que al evaluar en el sistema, da un resultado lo más cerca posible al original.

A $\mathbf{x}^*$ se le llama solución por mínimos cuadrados.

Y esta formulación es equivalente a encontrar el vector que minimice la distancia al cuadrado de $b$ y $b^*$ ($\underset{\mathbf{x}}{\operatorname{argmin}} ||A\mathbf{x}-b||^2$)

La regresión lineal es uno de estos sistemas, por lo que la solución no será exacta.

### Ecuación Normal 2

Se puede expresar el error cuadrático medio de forma matricial, además, para enccontrar el mínimo de esta función podemos prescindir del término $\frac{1}{m}$: `mse3`

--------------------------------------

Recordemos que $\mathbf{\hat{y}} = X\hat{\beta}$

**Y de esta manera se ve que el error cuadrático medio es en realidad la función objetivo de mínimos cuadrados (Escribir la ecuación de mínimos cuadrados con $\beta$** 

------------------------------------------------------

Para desarrollar esta expresión

Primero se aplican propiedades de la traspuesta

Se distribuyen el primer ... y segundo término

Y notemos que los valores marcados dan como resultado un número, por lo que se pueden restar

`Grafica`

Veamos que la función de error forma una superficie donde los ejes x y y representan los valores de los parámetros, y z el error, esta superficie es convexa, es decir que tiene un único punto mínimo.

Para encontrarlo, se deriva la expresión con respecto a $\beta$

Y se la iguala a 0, técnica común para encontrar mínimos.

Despejando el vector beta

Obtenemos la ecuación normal, este es el vector que evaluado en el sistema produce un resultado lo más cercano posible al esperado.  

Y desde la perspectiva de minimización, es el punto de menor error posible.

### Ecuación Normal 3

Volviendo a la gráfica del inicio

Escribimos las coordenadas matricialmente como desarrollamos hasta ahora

Si resolvemos la ecuación anterior

Obtenemos el vector de parámetros

Graficando la recta con intercepto 17.17 y pendiente 0.58 se grafica el ajuste obtenido.

### Interpretación de la regresión

Podemos preguntarnos entonces, si la regresión es útil para predecir valores desconocidos, por qué esta predicción se ve distinta, a los valores reales de los puntos que ya conocemos

En la realidad no todas las personas del grupo con un sueldo $x_i$, van a gastar lo mismo, es por esto que si tuvieramos la información completa de toda una población, tendríamos una gráfica de este estilo.

Con distintos consumos correspondiendo al mismo ingreso.

Si se calcula el promedio para cada $x_i$, tenemos una media de gasto de los consumos por grupo, denotada por los puntos naranjas.

A este promedio por grupos se le llama la esperanza de $y_i$ dado $x_i$.

Podemos trazar una línea que pase por cada promedio, esta es la llamada regresión lineal poblacional

Sin embargo, en problemas reales, generalmente se tiene sólo un punto correspondiendo a cada $x_i$, como en el ejemplo visto hasta ahora. 

Al ajustar la regresión lineal estamos tratando de estimar esta esperanza condicional para cada valor de $x$, en base a un fragmento o muestra de la población.

Inevitablemente, la estimación no será perfecta, debido a la pérdida de información al trabajar con un subconjunto de puntos.

Entonces, se puede decir que el interés está en predecir un valor lo más cercano posible al **promedio real** de la variable dependiente, *en nuestro ejemplo el gasto semanal*.

> Parámetros

Por otra parte, $\hat{\beta}_2$ es la pendiente, y nos dice cuánto cambia en el gasto semanal al incrementar el ingreso en 1.

Como cada predicción es una estimación del valor esperado de y dado x. Cuando $x=0$, $y=\hat{\beta}_1$.

Por esta razón el intercepto sólo tendrá sentido interpretarlo si en algún momento se puede tener que $x$ vale cero, lo cual en nuesto ejemplo no sirve ya que nadie puede gastar si no recibe dinero

> Notar que correlación no implica causalidad, además que en el análisis de correlación no se hace distinción de las variables, ambas se asumen como aleatorias, mientras que en regresión se asume que sólo la dependiente es aleatoria y las explicativas son fijas, no estocásticas.  

**HASTA AQUÍ: 8:44**

### Regresión Lineal Múltiple

A pesar de todo este desarrollo, en la vida real, existen valores que deseamos estimar y dependen de muchas variables, ¿cómo se ajusta una regresión lineal en estos casos?

Volviendo al modelo, consideremos ahora un atributo más de la persona. La cantidad de años de estudio, de manera general, mientras más especializada la persona, mejor sueldo podría tener.

Se reescribe la ecuación mantiendo el número de observación $i$, pero se le llama $x_2$ a la primer característica y $x_3$ a la segunda, sobreentendiendo que $x_1=1$ , esta nueva variable $x_3$ agrega un parámetro $\beta_3$ al modelo.

La esperanza condicional de $y_i$ depende ahora de los valores de $x_{i2}$ y $x_{i3}$.

Es decir, si antes se consideraba un grupo a todos los valores de $y_i$ en la línea vertical correspondiente a $x_i$, ahora, el grupo está dado por todos los valores $y_i$, sobre los planos perpendiculares que se cruzan en el punto dado por el valor de ambas variables.

Graficando la población, observamos que ahora tenemos variación también en la nueva dimensión.

Y si marcamos las esperanzas condicionales para cada grupo como un punto naranja, se puede trazar un plano que pase por todos las medias, esta es la, entre comillas, línea de regresión poblacional, ya que en realidad es la extensión a 2D de la recta, un plano

De manera similar al caso con una variable, se puede re escribir la estimación de forma matricial, así, el desarrollo para encontrar el vector de betas es idéntico, y la ecuación normal, sirve para una cantidad de variables arbitraria, si se considera cada una, como colúmna de la matriz de diseño, X.

Aplcando todo este desarrollo teórico, se puede calcular los parámetros estimados de muestra original, y se obtiene un plano ajustado.

En conclusion, una vez se tienen los $n$ parámetros ya calculados, se obtiene el valor estimado de la esperanza condicional de $y$, al organizar una nueva observación, con valores $a_1$ hasta $a_{n-1}$, como una fila en el formato de la matriz de diseño, y multiplicarlo por el vector de parámetros.

> Estandarización se deja para el método iterativo
