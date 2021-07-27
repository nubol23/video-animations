from manimlib import *
import numpy as np
from typing import Tuple


def normal_formula(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y


def create_axes(x_range, y_range, height=6, width=10):
    axes = Axes(x_range=x_range,
                y_range=y_range,
                height=height,
                width=width,
                y_axis_label='y',
                axis_config={'stroke_color': GREY_A,
                             'stroke_width': 2}, )
    axes.add_coordinate_labels(font_size=20)
                               # num_decimal_places=1)
    return axes


class Intro(Scene):
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.random.choice(np.linspace(0, 50, 100), 50, replace=False).reshape(-1, 1)
        Y = 4 + 13 * X + np.random.randn(*X.shape) * 30

        X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
        Y = (Y - np.mean(Y, axis=0, keepdims=True)) / np.std(Y, axis=0, keepdims=True)
        X -= X.min()
        Y -= Y.min()
        Y += 0.25
        X += 0.25
        phi_x = np.hstack((np.ones((len(X), 1)), X))

        return X, Y, phi_x

    def construct(self):
        axes = create_axes((0, 300, 40), (0, 200, 40))

        # Seguro que alguna vez viste un gráfico...
        X, Y, ΦX = np.load('Data/X_sample.npy'), np.load('Data/Y_sample.npy'), np.load('Data/phi_x_sample.npy')
        dots = [Dot(color=TEAL) for _ in range(len(X))]
        for dot, x, y in zip(dots, X.ravel(), Y.ravel()):
            dot.move_to(axes.c2p(x, y))
        dots_g = VGroup(*dots)
        self.play(FadeIn(axes), FadeIn(dots_g), run_time=2)
        self.wait(4)

        # Supongamos que en el eje x...
        y_label = axes.get_y_axis_label(r"\text{Gasto de consumo semanal}", direction=RIGHT,
                                        buff=0.5)
        x_label = axes.get_x_axis_label(r"\text{Ingreso semanal}", direction=DOWN)
        x_label.scale(0.8).next_to(axes, DOWN)
        y_label.scale(0.8).next_to(axes, LEFT).shift(RIGHT*2).rotate(PI/2)
        self.play(Write(x_label), run_time=2)
        self.wait(2)
        self.play(Write(y_label), run_time=2)
        self.wait(2)

        # En otras palabras cada punto representa...
        self.wait(7)

        # Digamos que quieres saber cuánto consume...
        dot4 = Dot(color=YELLOW_C)
        dot4.move_to(axes.c2p(280, 0))
        self.play(ShowCreation(dot4), run_time=1)

        dot44 = Dot()
        dot44.move_to(axes.c2p(280, 178))
        v_line = always_redraw(lambda: axes.get_v_line(dot44.get_bottom()))
        self.play(ShowCreation(v_line), run_time=1)
        self.wait(4)
        # Cómo podemos conocer ese valor?
        question = Tex(r'\hat{y}=?', color=YELLOW_C)
        question.next_to(dot44)
        self.play(Write(question), run_time=1)
        self.wait(3)

        # Pues este tipo de problemas de predicción...
        names = Text('Gauss y Legendre')
        names.to_corner(UR)
        self.play(Write(names), run_time=2)
        self.wait(6)

        self.play(FadeOut(question), FadeOut(v_line), FadeOut(dot4), run_time=1)
        self.wait(3)

        θ = normal_formula(ΦX, Y)
        best_fit_graph = axes.get_graph(lambda x: θ[0] + x*θ[1],
                                        color=RED_C)
        self.play(FadeOut(names), ShowCreation(best_fit_graph), run_time=1)
        # self.wait(5)
        self.wait(7)

        # La pregunta ahora es...
        angles = [30, -45, -10, 15, 25, -15]
        for angle in angles:
            self.play(Rotate(best_fit_graph, angle*DEGREES), run_time=1)
        self.wait()


class Rectas1(Scene):
    def construct(self):
        # "Empecemos por comprender la ecuación de una recta..."
        axes = create_axes((-2, 2.25, 0.5), (-2, 2.25, 0.5))

        Δy, Δx, b = 1, 2, ValueTracker(0)
        m = ValueTracker(Δy/Δx)

        line1 = Tex('y = ','b','+', 'm', 'x')
        self.play(FadeIn(line1), run_time=1)
        self.wait(4)

        # La cual está definida por su pendiente...
        self.play(line1.animate.set_color_by_tex_to_color_map({'m': YELLOW_C}), run_time=1)
        self.wait(0.5)
        self.play(line1.animate.set_color_by_tex_to_color_map({'b': YELLOW_C, 'm': WHITE}), run_time=1)
        # self.wait(1)
        self.play(line1.animate.set_color_by_tex_to_color_map({'b': WHITE})
                  .to_corner(UL).scale(0.8), FadeIn(axes), run_time=1)
        # self.play(line1.animate.to_corner(UL).scale(0.8), FadeIn(axes))

        # Donde la pendiente indica la inclinación...
        line_plot = axes.get_graph(lambda x: x*m.get_value() + b.get_value(), color=RED_C)
        dot = Dot(axes.c2p(Δx, m.get_value()*Δx+b.get_value()))
        dot_base = Dot(axes.c2p(Δx, 0))
        h_line = always_redraw(lambda: axes.get_h_line(dot_base.get_left(), color=BLUE_C))
        v_line = always_redraw(lambda: axes.get_v_line(dot.get_bottom(), color=BLUE_C))
        # *Creando la línea*
        self.play(ShowCreation(line_plot), run_time=1)
        self.wait(1)
        self.play(ShowCreation(h_line), ShowCreation(v_line), run_time=1)
        self.wait(3)
        # *Creando los textos de la pendiente*
        text_Δx = Tex(rf'\scriptstyle \Delta x={int(Δx)}', color=BLUE_C)
        text_Δx.next_to(h_line, UP, buff=SMALL_BUFF)
        text_Δy = Tex(rf'\scriptstyle \Delta y={int(m.get_value()*Δx)}', color=BLUE_C)
        text_Δy.next_to(v_line, RIGHT, buff=SMALL_BUFF)
        # line2 = Tex(rf'm = \frac{{\Delta y}}{{\Delta x}} = \frac{{{m.get_value()*Δx}}}{{{Δx}}} = {m.get_value()}')
        line2 = Tex(r'm = \frac{\Delta y}{\Delta x} = \frac{1}{2}')
        line2.scale(0.6)
        line2.next_to(line1, DOWN)
        self.play(FadeIn(text_Δy), run_time=1)
        self.wait(1)
        self.play(FadeIn(text_Δx), run_time=1)
        self.wait(1)
        self.play(Write(line2), run_time=1)
        self.wait(2)

        # *Moviendo la recta*
        # y el intercepto, en qué punto toca el eje y
        line_plot.add_updater(lambda mobj: mobj.become(axes.get_graph(lambda x: x*m.get_value()+b.get_value(), color=RED_C)))
        self.play(ApplyMethod(b.increment_value, 0.5),
                  FadeOut(text_Δx), FadeOut(text_Δy), FadeOut(h_line), FadeOut(v_line), run_time=2)
        # self.wait(2)

        dot = Dot(color=YELLOW_C)
        dot.move_to(axes.c2p(0, b.get_value()))
        text_dot = Tex(str(b.get_value()), color=YELLOW_C)
        text_dot.scale(0.5)
        text_dot.next_to(dot, RIGHT)
        line3 = Tex('b', '=', f'{b.get_value()}', r'\hspace{6pt}', 'm', '=', f'{m.get_value()}')
        line3.next_to(line1, DOWN)
        line3.scale(0.6)
        self.play(FadeIn(dot), Write(text_dot), ReplacementTransform(line2, line3), run_time=1)
        self.wait(4)

        # En la literatura estadística...
        # line1n = Tex(r'y = \beta_1+x\beta_2')
        line1n = Tex(r'y = ', r'\beta_1', '+', r'\beta_2', 'x')
        line1n.scale(0.8).to_corner(UL)
        # line3n = Tex(rf'\beta_1 = {b.get_value()}\hspace{{6pt}}\beta_2 = {m.get_value()}')
        line3n = Tex(r'\beta_1', '=', f'{b.get_value()}', r'\hspace{6pt}', r'\beta_2', '=', f'{m.get_value()}')
        line3n.scale(0.6).next_to(line1n, DOWN)

        self.play(FadeOut(dot), FadeOut(text_dot),
                  TransformMatchingTex(line1, line1n), TransformMatchingTex(line3, line3n), run_time=1)
                  # ReplacementTransform(line1, line1n), ReplacementTransform(line3, line3n))
        self.wait(3)

        # Es variando estos dos parámetros...
        line_plot.add_updater(lambda mobj: mobj.become(axes.get_graph(lambda x: x * m.get_value() + b.get_value(), color=RED_C)))
        t1, b1, plus, b2, t2 = line4 = VGroup(Tex('y ='), DecimalNumber(b.get_value(), num_decimal_places=2),
                                            Tex('+'), DecimalNumber(m.get_value(), num_decimal_places=2), Tex('x'))
        line4.arrange(RIGHT, buff=-0.1).to_corner(UL)
        t1.scale(0.8)
        t2.scale(0.8)
        plus.scale(0.8)
        b1.scale(0.6)
        b2.scale(0.6)

        f_always(b1.set_value, b.get_value)
        f_always(b2.set_value, m.get_value)
        self.play(FadeOut(line3n), ReplacementTransform(line1n, line4))

        for inc_b, inc_m in np.random.uniform(-1, 1, (6, 2)):
            self.play(ApplyMethod(b.increment_value, inc_b), ApplyMethod(m.increment_value, inc_m), run_time=1)

def align_points(p1, p2):
    if p2[1] - p1[1] > 0:
        # la línea está arriba
        p2[1] += 0.08
    else:
        p2[1] += 0.1
        p1[1] -= 0.1
    return p1, p2


class Rectas2(Scene):
    def construct(self):
        buff = 0.7

        run_time = 1

        # Ya conocemos las coordenadas de alguinos puntos...
        data_points = Tex('(', 'x_i', r'\hspace{2pt},', 'y_i', ')')
        self.play(Write(data_points), run_time=1)
        self.wait(6)

        # ...recibe el i-esimo valor en x...
        self.play(data_points.animate.set_color_by_tex_to_color_map({'x_i': YELLOW_C}), run_time=1)
        self.wait(1)
        self.play(data_points.animate.set_color_by_tex_to_color_map({'x_i': WHITE, 'y_i': YELLOW_C}), run_time=1)
        self.wait(1)

        self.play(data_points.animate.set_color_by_tex_to_color_map({'y_i': WHITE}).move_to(UP+RIGHT*4).shift(UP),
                  run_time=1) #12

        # Esta función se define asumiendo que tienen...
        line_eq1 = Tex('y', '=', r'\beta_1', '+', r'\beta_2', 'x')
        line_eq1.next_to(data_points, DOWN, buff=buff)
        self.play(Write(line_eq1), run_time=2)
        self.wait(4)

        # A esto se le agrega un término de error...
        line_eq2 = Tex('y', '=', r'\beta_1', '+', r'\beta_2', 'x', r'+', r'\varepsilon_i')
        line_eq2.next_to(data_points, DOWN, buff=buff)
        self.play(TransformMatchingTex(line_eq1, line_eq2), run_time=2)


        X, Y, ΦX = np.load('X_small.npy'), np.load('Y_small.npy'), np.load('phi_x_small.npy')
        axes = create_axes((0, 9, 2), (0, 9, 2), width=8)
        axes.move_to(LEFT*3).scale(0.9)
        dots = VGroup(*[Dot(axes.c2p(x, y), color=TEAL) for x, y in zip(X.ravel(), Y.ravel())])

        β = np.linalg.inv(ΦX.T @ ΦX) @ ΦX.T @ Y
        best_fit_graph = axes.get_graph(lambda x: β[0, 0] + x * β[1, 0], color=RED_C)
        dots_line = VGroup(*[Dot(axes.c2p(x, y)) for x, y in zip(X.ravel(), (ΦX@β).ravel())])
        lines = VGroup(*[DashedLine(*align_points(dot.get_top(), dot_line.get_bottom()), color=BLUE_C)
                         for dot, dot_line in zip(dots, dots_line)])

        graph = Group(axes, lines, dots, best_fit_graph)
        self.play(FadeIn(graph),
                  line_eq2.animate.set_color_by_tex_to_color_map({r'\varepsilon_i': BLUE_C}),
                  run_time=2)

        # En nuestro ejemplo el gasto mensual no solamente depende del ingreso...
        self.wait(5)
        self.wait(5)

        # Ahora queremos predecir el consumo para valores desconocidos pero que provienen de la misma distribución
        self.wait(1) # 40

        # Para estas observaciones nuevas no podemos calcular el término de error...
        line_eq3 = Tex(r'\hat{y}_i', '=', r'\hat{\beta}_1', '+', r'\hat{\beta}_2', 'x_i')
        line_eq3.next_to(line_eq2, DOWN, buff=buff)
        self.play(Write(line_eq3), run_time=2)
        self.wait(6)
        self.wait(2.5) # 53

        # A esta estimación la llamaremos \hat{y}
        self.play(line_eq3.animate.set_color_by_tex_to_color_map({r'\hat{y}': YELLOW_C}), run_time=run_time)
        self.wait(2.5)
        # ...además asumiremos que nuestros puntos son una muestra...
        self.play(line_eq3.animate.set_color_by_tex_to_color_map({r'\hat{y}': WHITE}), run_time=run_time)
        self.wait(3.5)
        self.play(line_eq3.animate.set_color_by_tex_to_color_map({r'\hat{\beta}_1': BLUE_C,
                                                                  r'\hat{\beta}_2': BLUE_C}),
                  run_time=1)
        self.wait(4)

        # Necesitamos una forma de medir dados los...
        self.play(FadeOut(data_points), FadeOut(line_eq2), FadeOut(graph),
                  line_eq3.animate.move_to(ORIGIN+UP)
                  .set_color_by_tex_to_color_map({r'\hat{\beta}_1': WHITE, r'\hat{\beta}_2': WHITE}),
                  run_time=1)
        self.wait(6)
        self.wait(1)


        # Esto se hace restando los valores...
        line_eq4_pre = Tex(r'\mathcal{L} = (\hat{y}_i - y_i)')
        line_eq4_pre.next_to(line_eq3, DOWN, buff=buff)
        self.play(Write(line_eq4_pre),
                  run_time=run_time)
        self.wait(3.5)

        # ...se eleva al cuadrado
        line_eq4 = Tex(r'\mathcal{L} = (\hat{y}_i - y_i)', '^2')
        line_eq4.next_to(line_eq3, DOWN, buff=buff)
        self.play(TransformMatchingTex(line_eq4_pre, line_eq4), run_time=1)
        self.wait(3)

        # Promediando estos valores...
        lse = Tex(r'J(\hat{\beta})=\frac{1}{m}\sum_{i=1}^m (\hat{y}_i-y_i)^2')
        lse.next_to(line_eq4, DOWN, buff=buff)
        self.play(Write(lse), run_time=2)
        self.wait(5)

        # Una línea se ajusta mejor a los puntos mientras menor el valor de J
        self.play(FadeOut(line_eq4), lse.animate.shift(UP), run_time=1)
        self.wait(5)
        self.wait(6)

        # ... menor valor posible de J(\beta)
        lse2 = Tex(r'\underset{\hat{\beta}_1,\hat{\beta}_2}{\operatorname{argmin}}\hspace{2pt}', r'J(\hat{\beta})=\frac{1}{m}\sum_{i=1}^m (\hat{y}_i-y_i)^2')
        lse2.move_to(lse)
        self.play(TransformMatchingTex(lse, lse2), run_time=2)

        # Esta operación se llama argumento del mínimo
        self.wait(5)
