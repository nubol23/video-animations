from manimlib import *
import numpy as np
from typing import Tuple


def normal_formula(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y


def create_axes(x_range, y_range):
    axes = Axes(x_range=x_range,
                y_range=y_range,
                height=6,
                width=10,
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
        # axes = create_axes((0, 4.25, 0.5), (0, 4.25, 0.5))
        axes = create_axes((0, 400, 50), (0, 300, 50))

        # Creando los puntos
        X, Y, ΦX = np.load('X.npy'), np.load('Y.npy'), np.load('phi_x.npy')
        dots = [Dot(color=TEAL) for _ in range(len(X))]
        for dot, x, y in zip(dots, X.ravel(), Y.ravel()):
            dot.move_to(axes.c2p(x, y))
        dots_g = VGroup(*dots)
        self.play(Write(axes), FadeIn(dots_g))
        # self.wait(6)

        # Labels "por ejemplo, si decimos que x representa el area..."
        # y_label = axes.get_y_axis_label(r"\scriptsize\text{precio }{\scriptstyle \times10^5}", direction=RIGHT,
        #                                 buff=0.5)
        # x_label = axes.get_x_axis_label(r"\scriptsize\text{area en }{\scriptstyle m^2\times10^3}", direction=DOWN)
        y_label = axes.get_y_axis_label(r"\text{Gasto de consumo semanal}", direction=RIGHT,
                                        buff=0.5)
        x_label = axes.get_x_axis_label(r"\text{Ingreso semanal}", direction=DOWN)
        x_label.scale(0.8).next_to(axes, DOWN)
        y_label.scale(0.8).next_to(axes, LEFT).shift(RIGHT*2).rotate(PI/2)
        self.play(Write(x_label), run_time=2)
        # self.wait(4)
        self.play(Write(y_label), run_time=2)
        # self.wait(2)

        # Punto desconocido "Sería útil poder saber..."
        dot4 = Dot(color=YELLOW_C)
        # dot4.move_to(axes.c2p(4, 0))
        dot4.move_to(axes.c2p(350, 0))
        self.play(ShowCreation(dot4))
        self.wait()

        dot44 = Dot()
        # dot44.move_to(axes.c2p(4, 4))
        dot44.move_to(axes.c2p(350, 280))
        v_line = always_redraw(lambda: axes.get_v_line(dot44.get_bottom()))
        self.play(ShowCreation(v_line))
        self.wait(3)

        question = Tex('\\hat{y}=?', color=YELLOW_C)
        question.next_to(dot44)
        self.play(Write(question))
        self.wait(5)
        self.play(FadeOut(question), FadeOut(v_line), FadeOut(dot4))

        # La idea consiste en encontrar la línea de mejor ajuste a los puntos
        θ = normal_formula(ΦX, Y)
        # Y_hat = ΦX@θ
        best_fit_graph = axes.get_graph(lambda x: θ[0] + x*θ[1],
                                        color=RED_C)
        self.play(ShowCreation(best_fit_graph))
        self.wait(2)

        # "Pero cómo lo conseguimos?"
        # for _ in range(10):
        #     self.play(Rotate(best_fit_graph,
        #                      np.random.randint(-45, 45)*DEGREES), run_time=1)
        angles = [30, -45, -10, 15, 25, -15]
        for angle in angles:
            self.play(Rotate(best_fit_graph, angle*DEGREES), run_time=1)


class Rectas1(Scene):
    def construct(self):
        # "Recordemos que una recta está definida..."
        axes = create_axes((-2, 2.25, 0.5), (-2, 2.25, 0.5))

        # Δy, Δx, b = 1, 2, 0
        Δy, Δx, b = 1, 2, ValueTracker(0)
        m = ValueTracker(Δy/Δx)

        line1 = Tex('y = ','b','+', 'm', 'x')
        self.play(FadeIn(line1))
        self.wait(3)
        self.play(line1.animate.set_color_by_tex_to_color_map({'m': YELLOW_C}))
        self.wait(1)
        self.play(line1.animate.set_color_by_tex_to_color_map({'b': YELLOW_C, 'm': WHITE}))
        self.wait(1)
        self.play(line1.animate.set_color_by_tex_to_color_map({'b': WHITE}))
        self.play(line1.animate.to_corner(UL).scale(0.8), FadeIn(axes))

        # donde la pendiente...
        line_plot = axes.get_graph(lambda x: x*m.get_value() + b.get_value(), color=RED_C)
        dot = Dot()
        dot.move_to(axes.c2p(Δx, m.get_value()*Δx+b.get_value()))
        h_line = always_redraw(lambda: axes.get_h_line(dot.get_left()))
        v_line = always_redraw(lambda: axes.get_v_line(dot.get_bottom()))

        # Creando la línea
        self.play(ShowCreation(line_plot))
        self.play(ShowCreation(h_line), ShowCreation(v_line))
        # Creando los textos de la pendiente
        text_Δx = Tex(f'\\scriptstyle \Delta x={int(Δx)}')
        text_Δx.next_to(v_line, RIGHT)
        text_Δy = Tex(f'\\scriptstyle \Delta y={int(m.get_value()*Δx)}')
        text_Δy.next_to(h_line, UP)
        line2 = Tex(f'm = \\frac{{\Delta y}}{{\Delta x}} = \\frac{{{m.get_value()*Δx}}}{{{Δx}}} = {m.get_value()}')
        line2.scale(0.5)
        line2.next_to(line1, DOWN)
        self.play(FadeIn(text_Δx), FadeIn(text_Δy), Write(line2))
        self.wait(3)

        # Moviendo la recta
        line_plot.add_updater(lambda mobj: mobj.become(axes.get_graph(lambda x: x*m.get_value()+b.get_value(), color=RED_C)))
        self.play(ApplyMethod(b.increment_value, 0.5),
                  FadeOut(text_Δx), FadeOut(text_Δy), FadeOut(h_line), FadeOut(v_line))
        self.wait(1)

        dot = Dot(color=YELLOW_C)
        dot.move_to(axes.c2p(0, b.get_value()))
        text_dot = Tex(str(b.get_value()), color=YELLOW_C)
        text_dot.scale(0.5)
        text_dot.next_to(dot, RIGHT)
        line3 = Tex(f'b = {b.get_value()}\\hspace{{6pt}}m = {m.get_value()}')
        line3.next_to(line1, DOWN)
        line3.scale(0.6)
        self.play(FadeIn(dot), Write(text_dot), ReplacementTransform(line2, line3))
        self.wait(2)

        # En la literatura estadística...
        line1n = Tex('y = \\beta_1+x\\beta_2')
        line1n.scale(0.8).to_corner(UL)
        line3n = Tex(f'\\beta_1 = {b.get_value()}\\hspace{{6pt}}\\beta_2 = {m.get_value()}')
        line3n.scale(0.6).next_to(line1n, DOWN)

        self.play(FadeOut(dot), FadeOut(text_dot),
                  ReplacementTransform(line1, line1n), ReplacementTransform(line3, line3n))
        self.wait(3)

        # Variando estos dos valores...
        line_plot.add_updater(lambda mobj: mobj.become(axes.get_graph(lambda x: x * m.get_value() + b.get_value(), color=RED_C)))
        t1, b1, t2, b2 = line4 = VGroup(Text('y ='), DecimalNumber(b.get_value(), num_decimal_places=2),
                                      Text('+ x'), DecimalNumber(m.get_value(), num_decimal_places=2))
        line4.arrange(RIGHT).to_corner(UL)
        t1.scale(0.8)
        t2.scale(0.8)
        b1.scale(0.6)
        b2.scale(0.6)

        f_always(b1.set_value, b.get_value)
        f_always(b2.set_value, m.get_value)
        self.play(FadeOut(line3n), ReplacementTransform(line1n, line4))

        for inc_b, inc_m in np.random.uniform(-1, 1, (10, 2)):
            self.play(ApplyMethod(b.increment_value, inc_b), ApplyMethod(m.increment_value, inc_m))


class Rectas2(Scene):
    def construct(self):
        run_time = 2
        data_points = VGroup(Tex('x_i'), Tex('y_i'))
        data_points.arrange(RIGHT)
        self.play(Write(data_points[0]), run_time=run_time)
        # self.wait(1)
        self.play(Write(data_points[1]), run_time=run_time)
        # self.wait(1)

        self.play(data_points.animate.move_to(UP))

        line_eq1 = Tex('y', '=', r'\beta_1', '+', r'\beta_2', 'x')
        self.play(Write(line_eq1), run_time=run_time)
        self.wait(4)
        line_eq2 = Tex('y', '=', r'\beta_1', '+', r'\beta_2', 'x', r'+\varepsilon_i')
        self.play(TransformMatchingTex(line_eq1, line_eq2), run_time=run_time)
        self.wait(4)
        self.play(data_points.animate.move_to(UP*2), line_eq2.animate.move_to(UP))
        self.wait(4)
        # Para estas observaciones nuevas no podemos...
        line_eq3 = Tex('\hat{y}', '=', r'\beta_1', '+', r'\beta_2', 'x')
        self.play(TransformMatchingTex(line_eq2.copy(), line_eq3), run_time=run_time)
        self.wait(6)
        self.wait(5)
        self.play(line_eq3.animate.set_color_by_tex_to_color_map({'\hat{y}': YELLOW_C}))
        self.wait(1)
        self.play(line_eq3.animate.set_color_by_tex_to_color_map({'\hat{y}': WHITE}))

        # Así podemos definir una forma de medir...
        self.play(FadeOut(data_points), FadeOut(line_eq2))
        self.wait(3)
        self.play(line_eq3.animate.set_color_by_tex_to_color_map({'\\beta_1': BLUE_C}))
        # self.wait()
        self.play(line_eq3.animate.set_color_by_tex_to_color_map({'\\beta_2': BLUE_C}))
        self.wait(1)
        line_eq4 = Tex(r'\mathcal{L} = (\hat{y}_i - y_i)^2')
        line_eq4.move_to(DOWN)
        self.wait()
        self.play(Write(line_eq4), run_time=run_time)
        self.wait(2)

        # Promediando estos valores...
        self.play(line_eq3.animate.move_to(UP*2), line_eq4.animate.move_to(UP*0.8))
        # self.wait()
        lse = Tex(r'J(\beta)=\frac{1}{m}\sum_{i=1}^m (\hat{y}_i-y_i)^2')
        lse.move_to(DOWN)
        self.wait(1)
        self.play(Write(lse), run_time=4)
        self.wait(7)

        # Ahora sabemos que una linea ajusta...
        self.play(FadeOut(line_eq4))
        self.play(lse.animate.move_to(UP*0.2))
        self.wait(4)

        # necesitamos una manera automática de dado un conjunto de datos...
        lse2 = Tex(r'\underset{\beta_1,\beta_2}{\operatorname{argmin}}\hspace{2pt}', r'J(\beta)=\frac{1}{m}\sum_{i=1}^m (\hat{y}_i-y_i)^2')
        self.play(TransformMatchingTex(lse, lse2), run_time=run_time)
        self.wait(5)
