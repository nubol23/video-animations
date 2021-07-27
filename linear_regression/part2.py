from manimlib import *
# from manim import *
import numpy as np
from time import perf_counter


def create_axes(x_range, y_range, height=6, width=10):
    axes = Axes(x_range=x_range,
                y_range=y_range,
                height=height,
                width=width,
                axis_config={'stroke_color': GREY_A,
                             'stroke_width': 2}, )
    axes.add_coordinate_labels(font_size=20,)
                               # num_decimal_places=1)
    return axes


class NormalEq1(Scene):
    def construct(self):
        # Para resolver esto debemos nortar primero que todo sistema de ecuaciones
        sistema1 = Tex(r'\begin{cases}a_{11}x_1+a_{12}x_2+\dots+a_{1n}x_n=b_1\\ a_{21}x_1+a_{22}x_2+\dots+a_{2n}x_n=b_2\\\vdots\\a_{n1}x_1+a_{n2}x_2+\dots+a_{nn}x_n=b_n\end{cases}')
        self.play(FadeIn(sistema1), run_time=1)
        self.wait(5)

        # se puede reescribir en función de matrices.
        sistema2 = Tex(r'\begin{bmatrix}a_{11}, a_{12}, \dots, a_{1n}\\a_{21}, a_{22}, \dots, a_{2n}\\\vdots\\a_{n1}, a_{n2}, \dots, a_{nn}\end{bmatrix}\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}','=',r'\begin{bmatrix}b_1\\b_2\\\vdots\\b_n\end{bmatrix}')
        self.play(ReplacementTransform(sistema1, sistema2), run_time=1)
        self.wait(2)
        # self.play(FadeOut(sistema2), run_time=1)


        # Esta notación es útil para simplificar cálculos.

        # Como tenemos m...
        line_eq1 = Tex(r'\hat{y}_i', r'=', r'\hat{\beta}_1 + \hat{\beta}_2x_i')
        line_eq1.move_to(LEFT*3.5)
        eq_range = Tex(r'i = 1, 2, \dots, m')
        eq_range.next_to(line_eq1, DOWN).scale(0.8)
        self.play(sistema2.animate.move_to(RIGHT*2.5), Write(line_eq1), Write(eq_range), run_time=1)
        self.wait(3) # 12

        # Podemos acomodar...
        vector_line1 = Tex(r'\begin{bmatrix}\hat{y}_1\\\hat{y}_2\\\vdots\\\hat{y}_m\end{bmatrix}', r'=',
                           r'\begin{bmatrix}\hat{\beta}_1+\hat{\beta}_2x_1\\\hat{\beta}_1+\hat{\beta}_2x_2\\\vdots\\\hat{\beta}_1+\hat{\beta}_2x_m\end{bmatrix}')
        # self.play(FadeOut(eq_range), TransformMatchingTex(line_eq1, vector_line1), run_time=1)
        vector_line1.move_to(RIGHT*3.5)
        self.play(TransformMatchingTex(sistema2, vector_line1), run_time=1)
        self.wait(6) # 19

        # Ahora podemos reescribir la suma...
        vector_line2 = Tex(r'\begin{bmatrix}\hat{y}_1\\\hat{y}_2\\\vdots\\\hat{y}_m\end{bmatrix}', r'=',
                           r'\begin{bmatrix}\hat{\beta}_1\cdot1 + \hat{\beta}_2\cdot x_1\\\hat{\beta}_1\cdot1 + \hat{\beta}_2\cdot x_2\\'+
                           r'\vdots\\\hat{\beta}_1\cdot1 + \hat{\beta}_2\cdot x_m\end{bmatrix}')
        # self.play(TransformMatchingTex(vector_line1, vector_line2), run_time=1)
        vector_line2.move_to(RIGHT*3.5)
        self.play(TransformMatchingTex(vector_line1, vector_line2), run_time=1)
        self.wait(5) # 24

        # Recordando la definición...
        # self.play(vector_line2.animate.move_to(LEFT*3.5), run_time=1)
        self.play(FadeOut(line_eq1), FadeOut(eq_range), vector_line2.animate.move_to(LEFT*3.5), run_time=1)
        self.wait(3)
        vector_v = Tex(r'\mathbf{x}_i = \begin{bmatrix}1\\ x_i\end{bmatrix}')
        vector_u = Tex(r'\hat{\beta} = \begin{bmatrix}\hat{\beta}_1\\ \hat{\beta}_2\end{bmatrix}')
        vector_u.next_to(vector_v, RIGHT)
        vectors = VGroup(vector_v, vector_u)
        vectors.move_to(RIGHT*3+UP*1.5)
        self.play(Write(vector_v), run_time=1)
        # self.wait(1) # 30
        self.play(Write(vector_u), run_time=1)
        self.wait(2)

        prod1 = Tex(r'\langle \mathbf{x}_i, \hat{\beta}\rangle = 1\cdot \hat{\beta}_1 + x_i\cdot\hat{\beta}_2')
        prod1.next_to(vectors, DOWN*2)
        self.play(Write(prod1), run_time=1)
        self.wait(4) # 38

        # Entonces cada elemento...
        vector_line3 = Tex(r'\begin{bmatrix}\hat{y}_1\\\hat{y}_2\\\vdots\\\hat{y}_m\end{bmatrix}', r'=',
                           r'\begin{bmatrix}\langle\begin{bmatrix}1\\x_1\end{bmatrix},\begin{bmatrix}\hat{\beta}_1\\'+
                    r'\hat{\beta}_2\end{bmatrix}\rangle\\ \langle\begin{bmatrix}1\\x_2\end{bmatrix},\begin{bmatrix}\hat{\beta}_1\\'+
                    r'\hat{\beta}_2\end{bmatrix}\rangle\\\vdots\\\langle\begin{bmatrix}1\\'+
                    r'x_m\end{bmatrix},\begin{bmatrix}\hat{\beta}_1\\\hat{\beta}_2\end{bmatrix}\rangle\end{bmatrix}')
        vector_line3.move_to(LEFT*3)
        self.play(TransformMatchingTex(vector_line2, vector_line3), run_time=1)
        self.wait(3) # 43

        # Pero podemos reescribir...
        prod2 = Tex(r'\mathbf{x}_i^T', r'\hat{\beta}', r'= \begin{bmatrix}1, x_i\end{bmatrix}\begin{bmatrix}\hat{\beta}_1\\ \hat{\beta}_2\end{bmatrix}')
        prod2.next_to(prod1, DOWN*2)
        self.play(Write(prod2[0]), run_time=1)
        self.wait(4) # 48
        self.play(Write(prod2[1]), run_time=1)
        self.play(Write(prod2[2]), run_time=1) # 50

        # aplicamos esto al vector
        vector_line4 = Tex(r'\begin{bmatrix}\hat{y}_1\\\hat{y}_2\\\vdots\\\hat{y}_m\end{bmatrix}', r'=',
                           r'\begin{bmatrix}\mathbf{x}_1^T\hat{\beta}\\\mathbf{x}_2^T\hat{\beta}\\\vdots\\\mathbf{x}_m^T\hat{\beta}\end{bmatrix}')
        vector_line4.move_to(LEFT * 3)
        self.play(TransformMatchingTex(vector_line3, vector_line4), run_time=1)
        self.wait(6)
        self.wait(4) # 1:01

        # Hasta este punto ya simplificamos mucho la notación...
        self.play(FadeOut(prod1), FadeOut(prod2), run_time=1)
        # ...consideremos cada vector x_i^T
        vector_vt = Tex(r'\mathbf{x}_i', '^T', r'=', r'\begin{bmatrix}1, x_i\end{bmatrix}')
        vector_vt.next_to(vector_u, LEFT)
        vectors2 = VGroup(vector_vt, vector_u)
        self.play(TransformMatchingTex(vector_v, vector_vt), run_time=1)
        self.wait(4) # 1:06

        # De modo que tenemos X, llamada matriz de diseño
        x_mat = Tex(r'X = \begin{bmatrix}1,\hspace{2pt}x_1\\1,\hspace{2pt}x_2\\\vdots\\1,\hspace{2pt}x_m\end{bmatrix}')
        x_mat.next_to(vectors2, DOWN*1.5)
        self.play(Write(x_mat), run_time=1)
        self.wait(3)

        # Al multiplicarla por el vector \hat{\beta}...
        vector_line5 = Tex(r'\begin{bmatrix}\hat{y}_1\\\hat{y}_2\\\vdots\\\hat{y}_m\end{bmatrix}', r'=',
                           r'\begin{bmatrix}1,\hspace{2pt}x_1\\1,\hspace{2pt}x_2\\\vdots\\1,\hspace{2pt}x_m\end{bmatrix}\cdot\begin{bmatrix}\hat{\beta}_1\\\hat{\beta}_2\end{bmatrix}')
        vector_line5.move_to(LEFT * 3)
        self.play(TransformMatchingTex(vector_line4, vector_line5), run_time=1)
        delay = 1
        self.wait(2+delay) # 1:13

        # ... que se escribe matricialmente...
        # self.play(FadeOut(vectors2), FadeOut(x_mat), run_time=1)
        self.play(FadeOut(vectors2), FadeOut(x_mat), vector_line5.animate.move_to(LEFT * 2), run_time=1)
        vector_line6 = Tex(r'\text{ }\Longrightarrow\text{ }\mathbf{\hat{y}} = X\hat{\beta}')
        vector_line6.next_to(vector_line5)
        self.play(Write(vector_line6), run_time=1)
        self.wait(3)


def wait_for_input(print_time=True):
    start = perf_counter()
    input()
    end = perf_counter()
    if print_time:
        print(round(end-start, 1))


class MinimosCuadrados(Scene):
    def construct(self):
        sistema3 = Tex(r'A', r'\mathbf{x}', '=', r'\mathbf{b}')
        sistema3.move_to(UP*0.8)
        self.play(Write(sistema3), run_time=1)
        self.wait(6)
        self.wait(3.5)

        # Existen casos donde el sistema...
        sistema4 = Tex(r'A', r'\mathbf{x}', '^*', '=', r'\mathbf{b}')
        sistema4.move_to(UP*0.8)
        self.play(TransformMatchingTex(sistema3, sistema4), run_time=1)
        self.wait(3)

        sistema5 = Tex(r'A', r'\mathbf{x}', r'^*', '=', r'\mathbf{b}', r'^*')
        sistema5.move_to(UP*0.8)
        self.play(TransformMatchingTex(sistema4, sistema5), run_time=1)
        self.wait(5)
        self.wait(5)

        # Al vector \mathbf{x}^* se le llama solucion por mínimos cuadrados
        # Esta formulación es equivalente...
        lsqrt = Tex(r'\mathbf{x}^* = \underset{\mathbf{x}}{\operatorname{argmin}} ||A\mathbf{x}-b||^2')
        lsqrt.next_to(sistema5, DOWN, buff=MED_LARGE_BUFF)
        self.play(Write(lsqrt), run_time=1)
        self.wait(5)
        self.wait(5)


def align_to_equal(next_to_tex_obj, tex_obj, direction, pattern_next_to='=', pattern='=', buff=LARGE_BUFF):
    eq_idx = 0
    for i in range(len(tex_obj)):
        if tex_obj[i].tex_string == pattern:
            eq_idx = i

    next_to_eq_idx = 0
    for i in range(len(next_to_tex_obj)):
        if next_to_tex_obj[i].tex_string == pattern_next_to:
            next_to_eq_idx = i

    # align equals
    tex_obj[eq_idx].next_to(next_to_tex_obj[next_to_eq_idx],
                            direction, buff=buff)

    # align left
    for i in reversed(range(eq_idx)):
        tex_obj[i].next_to(tex_obj[i+1], LEFT)
    # align right
    for i in range(eq_idx+1, len(tex_obj)):
        tex_obj[i].next_to(tex_obj[i - 1], RIGHT)


class NormalEq2(ThreeDScene):
    def construct(self):
        mse1 = Tex(r'J(\hat{\beta})=', r'\frac{1}{m}', r'\sum_{i=1}^m (\hat{y}_i-y_i)^2')
        mse1.move_to(UP).fix_in_frame()
        self.add(mse1)
        # self.play(FadeIn(mse1))
        self.wait(2.5)

        # Ahora se puede reescribir...
        mse2 = Tex(r'J(\hat{\beta})', r'=', r'\frac{1}{m}',
                   r'(\mathbf{y} - ', r'\mathbf{\hat{y}}', r')^T\cdot (\mathbf{y} - ', r'\mathbf{\hat{y}}', r')')
        mse2.move_to(UP).fix_in_frame()
        self.play(TransformMatchingTex(mse1, mse2))
        self.wait(4.9)

        mse3 = Tex(r'J(\hat{\beta})', r'=', r'(\mathbf{y} - ', r'\mathbf{\hat{y}}', r')^T\cdot (\mathbf{y} - ', r'\mathbf{\hat{y}}', r')')
        mse3.move_to(UP).fix_in_frame()
        self.play(TransformMatchingTex(mse2, mse3))
        self.wait(2.5)

        # Recordemos que \mathbf{\hat{y}} = X\hat{\beta}
        mse3_dev = Tex(r'=', r'(\mathbf{y} - ', r'\mathbf{X}\hat{\beta}', r')^T\cdot (\mathbf{y} - ', r'\mathbf{X}\hat{\beta}', r')')
        align_to_equal(mse3, mse3_dev, DOWN)
        self.play(TransformMatchingTex(mse3.copy(), mse3_dev))
        self.wait(3.7)

        # Esta formulación es una reescritura...
        lsqr1 = Tex(r'=', r'||', r'\mathbf{X}\hat{\beta}', r'-b||^2')
        align_to_equal(mse3_dev, lsqr1, DOWN)
        self.play(TransformMatchingTex(mse3_dev.copy(), lsqr1))
        self.wait(4.5)

        # Desarrollando esta expresión tenemos...
        lines = VGroup(Tex(r'J(\hat{\beta})', '=', r'(\mathbf{y} - ', r'\mathbf{X}\hat{\beta}', r')^T\cdot (\mathbf{y} - ', r'\mathbf{X}\hat{\beta}', r')'),
                       Tex(r'=', r'(\mathbf{y}^T - \hat{\beta}^T\cdot\mathbf{X}^T)\cdot (\mathbf{y} - \mathbf{X}\cdot\hat{\beta})'),
                       Tex(r'=', r'\mathbf{y}^T\mathbf{y} - ', r'\mathbf{y}^T\cdot\mathbf{X}\cdot\hat{\beta}', '-',
                           r'\hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{y}', r'+ \hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{X}\cdot\hat{\beta}'),
                       Tex(r'=', r'\mathbf{y}^T\mathbf{y} - ', r'2\hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{y}', r' + \hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{X}\cdot\hat{\beta}'))
        lines.fix_in_frame()

        lines[0].to_corner(UL)
        self.play(FadeOut(mse3), FadeOut(lsqr1),
                  TransformMatchingTex(mse3_dev, lines[0]))
        self.wait(1.8)

        align_to_equal(lines[0], lines[1], DOWN)
        self.play(Write(lines[1]))
        self.wait(1.4)

        # escribiendo por partes la distribución de términos
        align_to_equal(lines[1], lines[2], DOWN)
        self.play(Write(lines[2][0]), Write(lines[2][1]), Write(lines[2][2]))
        self.play(Write(lines[2][3]), Write(lines[2][4]), Write(lines[2][5]))
        self.wait(1.1)

        # coloreando los términos equivalentes
        self.play(lines[2].animate.set_color_by_tex_to_color_map({
                      r'\mathbf{y}^T\cdot\mathbf{X}\cdot\hat{\beta}': YELLOW_C,
                      r'\hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{y}': YELLOW_C}))
        self.wait(2.2)

        lines[3].set_color_by_tex_to_color_map({r'2\hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{y}': YELLOW_C})
        align_to_equal(lines[2], lines[3], DOWN)
        self.play(Write(lines[3]))
        self.wait(0.7)

        # La función del error cuadrático medio forma...
        mse4 = Tex(r'J(\hat{\beta})', r'=', r'\mathbf{y}^T\mathbf{y} - ', r'2\hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{y}', r' + \hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{X}\cdot\hat{\beta}')
        mse4.to_corner(UL).fix_in_frame()

        self.play(FadeOut(lines), TransformMatchingTex(lines[-1], mse4))
        # self.wait(0.5)

        # Graficando el MSE
        axes = ThreeDAxes(x_range=[-1, 1, 1], y_range=[-1, 1, 1], z_range=[0, 1, 1],
                          height=4, width=4, depth=2,
                          axis_config={'stroke_color': GREY_A, 'stroke_width': 2})

        y_label = axes.get_y_axis_label(r"\hat{\beta}_2", direction=RIGHT, buff=0.5)
        x_label = axes.get_x_axis_label(r"\hat{\beta}_1", direction=DOWN)
        x_label.scale(0.5)
        y_label.scale(0.5)

        paraboloid = ParametricSurface(
            lambda u, v: np.array([
                axes.c2p(u, v, u**2+v**2)
            ]),
            u_range=(-1, 1), v_range=(-1, 1),
            opacity=0.5,
            color=TEAL_C,
            resolution = (10, 10)
        )
        mesh = SurfaceMesh(paraboloid)
        mesh.set_stroke(WHITE, 1, opacity=0.5)
        graph = Group(mesh, paraboloid)

        plot = Group(axes, graph, x_label, y_label)
        plot.move_to(DOWN*0.8+RIGHT*0.8)

        frame = self.camera.frame
        frame.set_euler_angles(theta=30 * DEGREES, phi=45 * DEGREES)
        self.play(FadeIn(plot))
        self.wait(7)
        self.play(Rotate(plot, PI), run_time=5.2)
        # self.wait(0.3)
        self.play(plot.animate.scale(0.8).move_to(RIGHT * 4.2 + UP*0.2))
        self.wait(0.7)

        # Por esta razón se puede derivar...
        lines2 = VGroup(Tex(r'\nabla_{\hat{\beta}}J(\hat{\beta})', r'=', '-', '2' ,r'\mathbf{X}^T\mathbf{y} + ', '2', r'\mathbf{X}^T\mathbf{X}', r'\hat{\beta}'),
                        Tex(r'0', '=', '-', r'\mathbf{X}^T\mathbf{y} + ', r'\mathbf{X}^T\mathbf{X}', r'\hat{\beta}'),
                        Tex(r'\mathbf{X}^T\mathbf{X}', r'\hat{\beta}', '=', r'\mathbf{X}^T\mathbf{y}'),
                        Tex(r'\hat{\beta}', '=', r'(', r'\mathbf{X}^T\mathbf{X}', ')^{-1}', r'\mathbf{X}^T\mathbf{y}'))
        lines2.to_corner(UL).fix_in_frame()

        # lines2[0].next_to(mse4, DOWN, buff=0.75).shift(LEFT*1.5)
        lines2[0].next_to(mse4, DOWN, buff=0.75)
        self.play(Write(lines2[0]))
        self.wait(1.1)

        # E igualarla a 0...
        align_to_equal(lines2[0], lines2[1], DOWN)
        self.play(TransformMatchingTex(lines2[0].copy(), lines2[1]))
        self.wait(2.4)

        # Desarrollando
        for i, t in zip(range(2, len(lines2)), [2.4, 0, 3]):
            align_to_equal(lines2[i-1], lines2[i], DOWN)
            self.play(TransformMatchingTex(lines2[i-1].copy(), lines2[i]))
            if t > 0:
                self.wait(t)
        self.wait(6)
        self.wait(3)

        min_dot = Dot(axes.c2p(0.1, 0.1, 0.1**2+0.1**2), color=YELLOW_C)
        text_dot = Tex(r'\hat{\beta}', color=YELLOW_C)
        text_dot.next_to(min_dot, OUT * 3) \
            .rotate(70 * DEGREES, axis=np.array((1, 0, 0))) \
            .rotate(20 * DEGREES, axis=np.array((0, 0, 1))) \
            .scale(0.6)
        self.play(FadeIn(min_dot), FadeIn(text_dot))
        self.wait(2)


class NormalEq2Timer(ThreeDScene):
    def construct(self):
        mse1 = Tex(r'J(\hat{\beta})=', r'\frac{1}{m}', r'\sum_{i=1}^m (\hat{y}_i-y_i)^2')
        mse1.move_to(UP).fix_in_frame()

        # Ahora se puede reescribir...
        mse2 = Tex(r'J(\hat{\beta})', r'=', r'\frac{1}{m}',
                   r'(\mathbf{y} - ', r'\mathbf{\hat{y}}', r')^T\cdot (\mathbf{y} - ', r'\mathbf{\hat{y}}', r')')
        mse2.move_to(UP).fix_in_frame()

        mse3 = Tex(r'J(\hat{\beta})', r'=', r'(\mathbf{y} - ', r'\mathbf{\hat{y}}', r')^T\cdot (\mathbf{y} - ',
                   r'\mathbf{\hat{y}}', r')')
        mse3.move_to(UP).fix_in_frame()

        # Recordemos que \mathbf{\hat{y}} = X\hat{\beta}
        mse3_dev = Tex(r'=', r'(\mathbf{y} - ', r'\mathbf{X}\hat{\beta}', r')^T\cdot (\mathbf{y} - ',
                       r'\mathbf{X}\hat{\beta}', r')')
        mse3_dev.fix_in_frame()
        align_to_equal(mse3, mse3_dev, DOWN)

        # Esta formulación es una reescritura...
        lsqr1 = Tex(r'=', r'||', r'\mathbf{X}\hat{\beta}', r'-\mathbf{y}||^2')
        lsqr1.fix_in_frame()
        align_to_equal(mse3_dev, lsqr1, DOWN)

        # Desarrollando esta expresión tenemos...
        lines = VGroup(
            Tex(r'J(\hat{\beta})', '=', r'(\mathbf{y} - ', r'\mathbf{X}\hat{\beta}', r')^T\cdot (\mathbf{y} - ',
                r'\mathbf{X}\hat{\beta}', r')'),
            Tex(r'=',
                r'(\mathbf{y}^T - \hat{\beta}^T\cdot\mathbf{X}^T)\cdot (\mathbf{y} - \mathbf{X}\cdot\hat{\beta})'),
            Tex(r'=', r'\mathbf{y}^T\mathbf{y} - ', r'\mathbf{y}^T\cdot\mathbf{X}\cdot\hat{\beta}', '-',
                r'\hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{y}',
                r'+ \hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{X}\cdot\hat{\beta}'),
            Tex(r'=', r'\mathbf{y}^T\mathbf{y} - ', r'2\hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{y}',
                r' + \hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{X}\cdot\hat{\beta}'))
        lines.fix_in_frame()

        lines[0].to_corner(UL)

        align_to_equal(lines[0], lines[1], DOWN)

        # escribiendo por partes la distribución de términos
        align_to_equal(lines[1], lines[2], DOWN)

        # coloreando los términos equivalentes

        lines[3].set_color_by_tex_to_color_map({r'2\hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{y}': YELLOW_C})
        align_to_equal(lines[2], lines[3], DOWN)

        # La función del error cuadrático medio forma...
        mse4 = Tex(r'J(\hat{\beta})', r'=', r'\mathbf{y}^T\mathbf{y} - ',
                   r'2\hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{y}',
                   r' + \hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{X}\cdot\hat{\beta}')
        mse4.to_corner(UL).fix_in_frame()

        # Graficando el MSE
        axes = ThreeDAxes(x_range=[-1, 1, 1], y_range=[-1, 1, 1], z_range=[0, 1, 1],
                          height=4, width=4, depth=2,
                          axis_config={'stroke_color': GREY_A, 'stroke_width': 2})

        y_label = axes.get_y_axis_label(r"\hat{\beta}_2", direction=RIGHT, buff=0.5)
        x_label = axes.get_x_axis_label(r"\hat{\beta}_1", direction=DOWN)
        x_label.scale(0.5)
        y_label.scale(0.5)

        paraboloid = ParametricSurface(
            lambda u, v: np.array([
                axes.c2p(u, v, u ** 2 + v ** 2)
            ]),
            u_range=(-1, 1), v_range=(-1, 1),
            opacity=0.5,
            color=TEAL_C,
            resolution=(10, 10)
        )
        mesh = SurfaceMesh(paraboloid)
        mesh.set_stroke(WHITE, 1, opacity=0.5)
        graph = Group(mesh, paraboloid)

        plot = Group(axes, graph, x_label, y_label)
        plot.move_to(DOWN * 0.8 + RIGHT * 0.8)

        frame = self.camera.frame
        frame.set_euler_angles(theta=30 * DEGREES, phi=45 * DEGREES)

        # Por esta razón se puede derivar...
        lines2 = VGroup(
            Tex(r'\nabla_{\hat{\beta}}J(\hat{\beta})', r'=', '-', '2', r'\mathbf{X}^T\mathbf{y} + ', '2',
                r'\mathbf{X}^T\mathbf{X}', r'\hat{\beta}'),
            Tex(r'0', '=', '-', r'\mathbf{X}^T\mathbf{y} + ', r'\mathbf{X}^T\mathbf{X}', r'\hat{\beta}'),
            Tex(r'\mathbf{X}^T\mathbf{X}', r'\hat{\beta}', '=', r'\mathbf{X}^T\mathbf{y}'),
            Tex(r'\hat{\beta}', '=', r'(', r'\mathbf{X}^T\mathbf{X}', ')^{-1}', r'\mathbf{X}^T\mathbf{y}'))
        lines2.to_corner(UL).fix_in_frame()

        # lines2[0].next_to(mse4, DOWN, buff=0.75).shift(LEFT*1.5)
        lines2[0].next_to(mse4, DOWN, buff=0.75)

        # E igualarla a 0...
        align_to_equal(lines2[0], lines2[1], DOWN)
        # Desarrollando

        print('Listo...')
        wait_for_input(print_time=False)
        self.play(FadeIn(mse1))
        wait_for_input()
        self.play(TransformMatchingTex(mse1, mse2))
        wait_for_input()
        self.play(TransformMatchingTex(mse2, mse3))
        wait_for_input()
        self.play(TransformMatchingTex(mse3.copy(), mse3_dev))
        wait_for_input()
        self.play(TransformMatchingTex(mse3_dev.copy(), lsqr1))
        wait_for_input()
        self.play(FadeOut(mse3), FadeOut(lsqr1),
                  TransformMatchingTex(mse3_dev, lines[0]))
        wait_for_input()
        self.play(Write(lines[1]))
        wait_for_input()
        self.play(Write(lines[2][0]), Write(lines[2][1]), Write(lines[2][2]))
        wait_for_input()
        self.play(Write(lines[2][3]), Write(lines[2][4]), Write(lines[2][5]))
        wait_for_input()
        self.play(lines[2].animate.set_color_by_tex_to_color_map({
            r'\mathbf{y}^T\cdot\mathbf{X}\cdot\hat{\beta}': YELLOW_C,
            r'\hat{\beta}^T\cdot\mathbf{X}^T\cdot\mathbf{y}': YELLOW_C}))
        wait_for_input()
        self.play(Write(lines[3]))
        wait_for_input()
        self.play(FadeOut(lines), TransformMatchingTex(lines[-1], mse4))
        wait_for_input()
        self.play(FadeIn(plot))
        wait_for_input()
        self.play(Rotate(plot, PI), run_time=5)
        wait_for_input()
        self.play(plot.animate.scale(0.8).move_to(RIGHT * 4.2 + UP * 0.2))
        wait_for_input()
        self.play(Write(lines2[0]))
        wait_for_input()
        self.play(TransformMatchingTex(lines2[0].copy(), lines2[1]))
        wait_for_input()
        for i in range(2, len(lines2)):
            align_to_equal(lines2[i - 1], lines2[i], DOWN)
            self.play(TransformMatchingTex(lines2[i - 1].copy(), lines2[i]))
            wait_for_input()

        min_dot = Dot(axes.c2p(0.1, 0.1, 0.1 ** 2 + 0.1 ** 2), color=YELLOW_C)
        text_dot = Tex(r'\hat{\beta}', color=YELLOW_C)
        text_dot.next_to(min_dot, OUT * 3) \
            .rotate(70 * DEGREES, axis=np.array((1, 0, 0))) \
            .rotate(20 * DEGREES, axis=np.array((0, 0, 1))) \
            .scale(0.6)
        self.play(FadeIn(min_dot), FadeIn(text_dot))
        wait_for_input()


def normal_formula(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y


class NormalEq3(Scene):
    def construct(self):
        # Regresando al conjunto de puntos del incio
        axes = create_axes((0, 300, 40), (0, 200, 40))

        X, Y, ΦX = np.load('Data/X_sample.npy'), np.load('Data/Y_sample.npy'), np.load('Data/phi_x_sample.npy')
        dots = VGroup(*[Dot(axes.c2p(x, y), color=TEAL) for x, y in zip(X.ravel(), Y.ravel())])
        self.play(FadeIn(axes), FadeIn(dots))
        self.wait(0.8)

        # Graficando la recta obtenida por estos...
        plot = Group(axes, dots)
        self.play(plot.animate.scale(0.7).to_corner(LEFT))

        x_mat = Tex(rf'\mathbf{{X}} = \begin{{bmatrix}}1,\hspace{{2pt}}{{{ΦX[0, 1]:.2f}}}\\1,\hspace{{2pt}}{{{ΦX[1, 1]:.2f}}}\\\vdots\\1,\hspace{{2pt}}{{{ΦX[-1, 1]:.2f}}}\end{{bmatrix}}')
        y_mat = Tex(rf'\mathbf{{y}} = \begin{{bmatrix}}{{{Y[0, 0]:.2f}}}\\ {{{Y[1, 0]:.2f}}}\\\vdots\\{{{Y[-1, 0]:.2f}}}\end{{bmatrix}}')

        mats = VGroup(x_mat, y_mat)
        mats.arrange(RIGHT).move_to(UP+RIGHT*4).scale(0.9)

        self.play(FadeIn(mats), run_time=2)
        self.wait(2)

        # Si se resuelve la ecuación anterior
        normal_eq = Tex(r'\hat{\beta}', '=', r'(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}')
        normal_eq.next_to(mats, DOWN, buff=LARGE_BUFF)
        self.play(Write(normal_eq))
        self.wait(1.6)

        # Se obtiene el vector de parámetros
        β = np.linalg.inv(ΦX.T @ ΦX) @ ΦX.T @ Y

        β_mat = Tex(r'\hat{\beta}', '=', rf'\begin{{bmatrix}}{{{β[0, 0]:.2f}}}\\ {{{β[1, 0]:.2f}}}\end{{bmatrix}}')
        β_mat.next_to(mats, DOWN, buff=LARGE_BUFF)

        self.play(TransformMatchingTex(normal_eq, β_mat))
        self.wait(1.6)

        # Graficando la recta obtenida...
        best_fit_graph = axes.get_graph(lambda x: β[0, 0] + x * β[1, 0],
                                        color=RED_C)
        self.play(ShowCreation(best_fit_graph))
        self.wait(4)

        new_plot = plot.deepcopy()
        new_plot.scale(1/0.7).move_to(ORIGIN)
        self.play(FadeOut(mats), FadeOut(β_mat), plot.animate.become(new_plot),
                  best_fit_graph.animate.become(new_plot[0].get_graph(lambda x: β[0, 0] + x * β[1, 0], color=RED_C)))
        self.wait(2.5)

        # Transicion entre escenas
        axes_new = create_axes((0, 300, 40), (0, 200, 40))
        axes_new.to_corner(LEFT).scale(0.9)
        y_label = axes_new.get_y_axis_label(r"\text{Gasto de consumo semanal}", direction=RIGHT,
                                        buff=0.5)
        x_label = axes_new.get_x_axis_label(r"\text{Ingreso semanal}", direction=DOWN)
        x_label.scale(0.8).next_to(axes_new, DOWN)
        y_label.scale(0.8).next_to(axes_new, LEFT).shift(RIGHT * 2).rotate(PI / 2)

        X_sample, Y_sample, ΦX_sample = np.load('Data/X_sample.npy'), np.load('Data/Y_sample.npy'), np.load(
            'Data/phi_x_sample.npy')
        β_sample = normal_formula(ΦX_sample, Y_sample)

        dots_sample = VGroup(*[Dot(axes_new.c2p(x, y), color=TEAL) for x, y in zip(X_sample.ravel(), Y_sample.ravel())])

        best_fit_graph_sample = axes_new.get_graph(lambda x: β_sample[0, 0] + x * β_sample[1, 0], color=RED_C)

        self.play(axes.animate.become(axes_new),
                  FadeIn(x_label), FadeIn(y_label),
                  dots.animate.become(dots_sample),
                  best_fit_graph.animate.become(best_fit_graph_sample))


class NormalEq3Timer(Scene):
    def construct(self):
        # Regresando al conjunto de puntos del incio
        axes = create_axes((0, 300, 40), (0, 200, 40))

        X, Y, ΦX = np.load('Data/X_sample.npy'), np.load('Data/Y_sample.npy'), np.load('Data/phi_x_sample.npy')
        dots = VGroup(*[Dot(axes.c2p(x, y), color=TEAL) for x, y in zip(X.ravel(), Y.ravel())])

        # Graficando la recta obtenida por estos...
        plot = Group(axes, dots)

        x_mat = Tex(rf'\mathbf{{X}} = \begin{{bmatrix}}1,\hspace{{2pt}}{{{ΦX[0, 1]:.2f}}}\\1,\hspace{{2pt}}{{{ΦX[1, 1]:.2f}}}\\\vdots\\1,\hspace{{2pt}}{{{ΦX[-1, 1]:.2f}}}\end{{bmatrix}}')
        y_mat = Tex(rf'\mathbf{{y}} = \begin{{bmatrix}}{{{Y[0, 0]:.2f}}}\\ {{{Y[1, 0]:.2f}}}\\\vdots\\{{{Y[-1, 0]:.2f}}}\end{{bmatrix}}')

        mats = VGroup(x_mat, y_mat)
        mats.arrange(RIGHT).move_to(UP+RIGHT*4).scale(0.9)

        # Si se resuelve la ecuación anterior
        normal_eq = Tex(r'\hat{\beta}', '=', r'(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}')
        normal_eq.next_to(mats, DOWN, buff=LARGE_BUFF)

        # Se obtiene el vector de parámetros
        β = np.linalg.inv(ΦX.T @ ΦX) @ ΦX.T @ Y

        β_mat = Tex(r'\hat{\beta}', '=', rf'\begin{{bmatrix}}{{{β[0, 0]:.2f}}}\\ {{{β[1, 0]:.2f}}}\end{{bmatrix}}')
        β_mat.next_to(mats, DOWN, buff=LARGE_BUFF)

        # Graficando la recta obtenida...
        best_fit_graph = axes.get_graph(lambda x: β[0, 0] + x * β[1, 0],
                                        color=RED_C)

        new_plot = plot.deepcopy()
        new_plot.scale(1/0.7).move_to(ORIGIN)

        wait_for_input(False)
        self.play(FadeIn(axes), FadeIn(dots))
        wait_for_input()
        self.play(plot.animate.scale(0.7).to_corner(LEFT))
        wait_for_input()
        self.play(Write(mats))
        wait_for_input()
        self.play(Write(normal_eq))
        wait_for_input()
        self.play(TransformMatchingTex(normal_eq, β_mat))
        wait_for_input()
        self.play(ShowCreation(best_fit_graph))
        wait_for_input()
        self.play(FadeOut(mats), FadeOut(β_mat), plot.animate.become(new_plot),
                  best_fit_graph.animate.become(new_plot[0].get_graph(lambda x: β[0, 0] + x * β[1, 0], color=RED_C)))
        wait_for_input()


class DataNorm(ThreeDScene):
    def calculate_error(self, u, v, phi_x, Y):
        params = np.array([[u], [v]])
        y_hat_space = phi_x @ params
        errors_space = 1 / len(Y) * np.sum((y_hat_space - Y) ** 2, axis=0)
        return float(errors_space)

    def construct(self):
        # X, Y = np.load('X.npy'), np.load('Y.npy')
        X, Y = np.load('Data/X_sample.npy'), np.load('Data/Y_sample.npy')

        X_tilde = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
        Y_tilde = (Y - np.mean(Y, axis=0, keepdims=True)) / np.std(Y, axis=0, keepdims=True)
        phi_x = np.hstack((np.ones((len(X_tilde), 1)), X_tilde))

        # Vamos a denotar la segunda colúmna...
        notation = Tex(r'\text{Colúmna 2: }X_{:, 2}')
        # notation.move_to(RIGHT*4+UP*3.5).fix_in_frame()
        notation.to_corner(UP).fix_in_frame()

        self.play(Write(notation))

        # Para la segunda colúmna se calcula su media y su std...
        mean = Tex(r'\mu_{X_{:, 2}}')
        std = Tex(r'\sigma_{X_{:, 2}}')
        est = VGroup(mean, std)
        est.arrange(RIGHT).next_to(notation, DOWN, buff=LARGE_BUFF).fix_in_frame()

        self.play(Write(est[0]))
        self.play(Write(est[1]))

        # Se realiza el cálculo elemento a elemento...
        standarization_x = Tex(r'\mathbf{\tilde{X}_{:, 2}}} = \frac{\mathbf{X}_{:, 2} - \mu_{\mathbf{X}_{:, 2}}}{\sigma_{\mathbf{X}_{:, 2}}}')
        standarization_x.scale(1).next_to(est, DOWN, buff=LARGE_BUFF).fix_in_frame()
        self.play(Write(standarization_x))

        # Aplicando el mismo procedimiento para y
        standarization_y = Tex(r'\mathbf{\tilde{y}} = \frac{\mathbf{y} - \mu_{\mathbf{y}}}{\sigma_{\mathbf{y}}}')
        standarization_y.scale(1).next_to(standarization_x, DOWN, buff=LARGE_BUFF).fix_in_frame()
        self.play(Write(standarization_y))
        standarization = VGroup(standarization_x, standarization_y)

        self.play(FadeOut(est), FadeOut(notation),
                  standarization.animate.scale(0.7).to_corner(UR))

        # Graficar puntos en escala real y llevarlos a la normal
        # axes2d_pre = create_axes((0, 400, 50), (0, 300, 50))
        axes2d_pre = create_axes((0, 300, 40), (0, 200, 40))
        dots_pre = VGroup(*[Dot(axes2d_pre.c2p(x, y), color=TEAL) for x, y in zip(X.ravel(), Y.ravel())])
        # for dot in dots_pre:
        #     dot.scale(0.6)

        plot2d_pre = Group(axes2d_pre, dots_pre)
        plot2d_pre.to_corner(LEFT).fix_in_frame()
        self.play(FadeIn(plot2d_pre))

        self.play(*[dot.animate.move_to(axes2d_pre.c2p(x, y)) for dot, x, y in zip(dots_pre, X_tilde.ravel(), Y_tilde.ravel())])

        # Zoom al grafico
        axes2d = create_axes((-2, 4, 1), (-2, 4, 1), 6, 8)
        dots = VGroup(*[Dot(axes2d.c2p(x, y), color=TEAL) for x, y in zip(X_tilde.ravel(), Y_tilde.ravel())])
        # for dot in dots:
        #     dot.scale(0.6)
        plot2d = Group(axes2d, dots)
        plot2d.to_corner(LEFT).fix_in_frame()

        self.play(plot2d_pre.animate.become(plot2d))

        β = np.linalg.inv(phi_x.T @ phi_x) @ phi_x.T @ Y_tilde
        error = round(1 / len(Y_tilde) * np.sum((phi_x @ β - Y_tilde) ** 2), 7)

        # Graficando Superficie de error
        axes3d = ThreeDAxes(x_range=[-5, 5, 0.5], y_range=[-5, 7, 0.5], z_range=[0, 60, 10],
                          height=4, width=4, depth=4,
                          axis_config={'stroke_color': GREY_A, 'stroke_width': 2})

        y_label = axes3d.get_y_axis_label(r"\hat{\beta}_2", direction=RIGHT, buff=0.5)
        x_label = axes3d.get_x_axis_label(r"\hat{\beta}_1", direction=DOWN)
        x_label.scale(0.5)
        y_label.scale(0.5)

        paraboloid = ParametricSurface(
            lambda u, v: np.array([
                axes3d.c2p(u, v, self.calculate_error(u, v, phi_x, Y_tilde))
            ]),
            u_range=(-4.5, 4.5), v_range=(-4.5, 6.5),
            opacity=0.5,
            color=TEAL_C,
            resolution=(10, 10)
        )
        mesh = SurfaceMesh(paraboloid)
        mesh.set_stroke(WHITE, 1, opacity=0.2)
        graph3d = Group(mesh, paraboloid)

        plot3d = Group(axes3d, graph3d, x_label, y_label)
        plot3d.scale(0.8)
        plot3d.move_to(RIGHT * 4.25 + UP*0.5)

        frame = self.camera.frame
        frame.set_euler_angles(theta=30 * DEGREES, phi=45 * DEGREES)
        self.play(FadeIn(plot3d))

        # min point
        min_dot = Dot(axes3d.c2p(β[0,0], β[1, 0], error), color=YELLOW_C)
        text_dot = Tex('(', r'\hat{\beta}_1', ',' r'\hat{\beta}_2', ',', r'J(\hat{\beta})', ')', color=YELLOW_C)
        text_dot.next_to(min_dot, OUT*3)\
            .rotate(70*DEGREES, axis=np.array((1, 0, 0)))\
            .rotate(20*DEGREES, axis=np.array((0, 0, 1)))\
            .scale(0.6)

        self.play(FadeIn(min_dot), FadeIn(text_dot))
        self.play(FadeOut(standarization), Rotate(plot3d, 2*PI), run_time=5)
        # self.wait(3)

        # Con los valores de los betas...
        text_dot_val = Tex('(', f'{β[0,0]:.2f}', ',' f'{β[1,0]:.2f}', ',', f'{error:.2f}', ')', color=YELLOW_C)
        text_dot_val.next_to(min_dot, OUT * 3) \
            .rotate(70 * DEGREES, axis=np.array((1, 0, 0))) \
            .rotate(20 * DEGREES, axis=np.array((0, 0, 1))) \
            .scale(0.6)

        β_mat = Tex(r'\hat{\beta}', '=', rf'\begin{{bmatrix}}{{{β[0, 0]:.2f}}}\\ {{{β[1, 0]:.2f}}}\end{{bmatrix}}')
        β_mat.next_to(plot3d, UP, buff=SMALL_BUFF).fix_in_frame()

        self.play(TransformMatchingTex(text_dot, text_dot_val), Write(β_mat))

        # Tenemos la línea de mejor ajuste...
        best_fit_graph = axes2d.get_graph(lambda x: β[0,0] + x*β[1,0], color=RED_C).fix_in_frame()
        self.play(ShowCreation(best_fit_graph))
        self.wait(4)


class Motivacion(Scene):
    def construct(self):
        pass
