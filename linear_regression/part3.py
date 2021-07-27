from manimlib import *
import numpy as np
from collections import Counter
from time import perf_counter


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


def normal_formula(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y


def create_legends():
    legend_population = VGroup(Line([0, 0, 0], [0 + 0.2, 0, 0], color=YELLOW_C),
                               Text('población', font_size=30, color=YELLOW_C))
    legend_population.arrange(RIGHT)

    eq_y = Tex(r'E(\mathbf{Y}|\mathbf{X})=\beta_1+\beta_2\mathbf{X}', color=YELLOW_C)
    eq_y.next_to(legend_population, DOWN, buff=0.25)

    legend_sample = VGroup(Line([0, 0, 0], [0 + 0.2, 0, 0], color=RED_C), Text('muestra', font_size=30, color=RED_C))
    legend_sample[0].next_to(legend_population[0], DOWN, buff=1+MED_LARGE_BUFF)
    legend_sample[1].next_to(legend_sample[0], RIGHT)

    eq_pred = Tex(r'\mathbf{\hat{Y}}=\hat{\beta}_1+\hat{\beta}_2\mathbf{X}', color=RED_C)
    eq_pred.next_to(legend_sample, DOWN, buff=0.25)

    legends = Group(legend_population, eq_y, legend_sample, eq_pred)

    return legends


class Interpretacion(Scene):
    def construct(self):
        axes = create_axes((0, 300, 40), (0, 200, 40))
        axes.to_corner(LEFT).scale(0.9)
        y_label = axes.get_y_axis_label(r"\text{Gasto de consumo semanal}", direction=RIGHT,
                                        buff=0.5)
        x_label = axes.get_x_axis_label(r"\text{Ingreso semanal}", direction=DOWN)
        x_label.scale(0.8).next_to(axes, DOWN)
        y_label.scale(0.8).next_to(axes, LEFT).shift(RIGHT * 2).rotate(PI / 2)

        X_sample, Y_sample, ΦX_sample = np.load('Data/X_sample.npy'), np.load('Data/Y_sample.npy'), np.load('Data/phi_x_sample.npy')
        β_sample = normal_formula(ΦX_sample, Y_sample)

        dots_sample = VGroup(*[Dot(axes.c2p(x, y), color=TEAL) for x, y in zip(X_sample.ravel(), Y_sample.ravel())])

        best_fit_graph_sample = axes.get_graph(lambda x: β_sample[0, 0] + x * β_sample[1, 0], color=RED_C)

        plot_sample_fit = Group(axes, x_label, y_label, dots_sample, best_fit_graph_sample)
        self.add(plot_sample_fit)
        self.wait(7)
        self.wait(5.3)

        # En la realidad el grupo de personas...
        self.play(FadeOut(best_fit_graph_sample))
        self.wait(5)
        self.wait(4.3)

        # tendríamos la siguiente gráfica
        X_population, Y_population = np.load('Data/X_population.npy'), np.load('Data/Y_population.npy')
        dots_population = VGroup(*[Dot(axes.c2p(x, y), color=TEAL) for x, y in zip(X_population.ravel(), Y_population.ravel())])
        for dot in dots_population:
            dot.scale(0.6)

        self.play(TransformFromCopy(dots_sample, dots_population), FadeOut(dots_sample))
        self.wait(3.5)

        # si se calcula el promedio de cada grupo
        X_mean, Y_mean = np.load('Data/X_mean.npy'), np.load('Data/Y_mean.npy')
        ΦX_mean = np.hstack((np.ones((len(X_mean), 1)), X_mean))
        dots_mean = VGroup(*[Dot(axes.c2p(x, y), color=ORANGE) for x, y in zip(X_mean.ravel(), Y_mean.ravel())])
        self.play(FadeIn(dots_mean))
        self.wait(8.8)

        # A este promedio por grupos se le llama...
        cond_exp = Tex(r'E(y_i|x_i)', color=ORANGE)
        # cond_exp.scale(0.9).to_corner(UR)
        cond_exp.scale(0.9).move_to(axes.c2p(40, 160))
        self.play(Write(cond_exp))
        self.wait(2)

        legends = create_legends()
        legends.move_to(RIGHT*5.3+UP)
        legends[1].scale(0.7)
        legends[3].scale(0.7)

        # Se puede trazar una línea que pase por cada promedio
        β_population = normal_formula(ΦX_mean, Y_mean)
        best_fit_graph_population = axes.get_graph(lambda x: β_population[0, 0] + x*β_population[1, 0], color=YELLOW_C)
        self.play(ShowCreation(best_fit_graph_population), FadeIn(legends[0]), FadeIn(legends[1]))
        self.wait(7)
        self.wait(6)

        # en base a un fragmento o muestra de la población
        self.play(ShowCreation(best_fit_graph_sample),
                  FadeIn(legends[2]), FadeIn(legends[3]))
        self.wait(8)
        self.wait(8.8)
        self.play(TransformFromCopy(dots_population, dots_sample), FadeOut(dots_population), FadeOut(legends),
                  FadeOut(best_fit_graph_population), FadeOut(dots_mean), FadeOut(cond_exp))
                  # plot_sample_fit.animate.move_to(ORIGIN))
        self.wait(5)
        self.wait(5.9)
        # notemos que \hat{\beta}_2 es la pendiente
        x_coord = 200
        y_coord = float(np.array([[1, x_coord]])@β_sample)
        dot_x = Dot(axes.c2p(x_coord, β_sample[0,0]))
        dot_y = Dot(axes.c2p(x_coord, y_coord))

        v_line = DashedLine(dot_x.get_top()-[0, 0.08, 0], dot_y.get_bottom(), color=BLUE_C, stroke_width=3)
        h_line = axes.get_h_line(dot_x.get_left() + [0.08, 0, 0], color=BLUE_C, stroke_width=3)
        text_Δx = Tex(rf'\scriptstyle \Delta x={x_coord}', color=BLUE_C)
        text_Δx.next_to(h_line, UP)
        text_Δy = Tex(rf'\scriptstyle \Delta y={y_coord-β_sample[0,0]:.2f}', color=BLUE_C)
        text_Δy.next_to(v_line, RIGHT)
        self.play(ShowCreation(v_line), ShowCreation(h_line), Write(text_Δy), Write(text_Δx))
        self.wait(1.7)
        beta_2 = Tex(rf'\hat{{\beta}}_2=\frac{{{y_coord-β_sample[0,0]:.2f}}}{{{x_coord}}}={β_sample[1,0]:.2f}',
                     color=BLUE_C)
        beta_2.scale(0.8).to_corner(UR)
        self.play(Write(beta_2))
        self.wait(3.5)
        # Como cada predicción es el valor esperado de y dado x
        pred1 = Tex(r'\hat{y}_i', '=', r'\hat{\beta}_1', '+', r'\hat{\beta}_2', 'x_i', color=YELLOW_C)
        pred1.next_to(beta_2, DOWN, buff=MED_LARGE_BUFF)
        self.play(Write(pred1))
        self.wait(4.6)
        # Cuando x = 0
        dot_pred = Dot(axes.c2p(0, 0), color=YELLOW_C)
        pred2 = Tex(r'\hat{y}_i', '=', r'\hat{\beta}_1', '+', r'\hat{\beta}_2', '0', color=YELLOW_C)
        pred2.next_to(beta_2, DOWN, buff=MED_LARGE_BUFF)
        self.play(TransformMatchingTex(pred1, pred2), ShowCreation(dot_pred))
        self.wait(0.9)

        # la estimación toma el valor de \hat{\beta}_1 (beta 1 sombrero)
        pred3 = Tex(r'\hat{y}_i', '=', r'\hat{\beta}_1', color=YELLOW_C)
        pred3.next_to(beta_2, DOWN, buff=MED_LARGE_BUFF)
        self.play(TransformMatchingTex(pred2, pred3),
                  dot_pred.animate.move_to(axes.c2p(0, β_sample[0,0])))
        self.wait(15.2)
        # Por esta razón el interepto sólo tendrá sentido...


def wait_for_input(print_time=True):
    start = perf_counter()
    input()
    end = perf_counter()
    if print_time:
        print(round(end-start, 1))


class InterpretacionTimer(Scene):
    def construct(self):
        axes = create_axes((0, 300, 40), (0, 200, 40))
        axes.to_corner(LEFT).scale(0.9)
        y_label = axes.get_y_axis_label(r"\text{Gasto de consumo semanal}", direction=RIGHT,
                                        buff=0.5)
        x_label = axes.get_x_axis_label(r"\text{Ingreso semanal}", direction=DOWN)
        x_label.scale(0.8).next_to(axes, DOWN)
        y_label.scale(0.8).next_to(axes, LEFT).shift(RIGHT * 2).rotate(PI / 2)

        X_sample, Y_sample, ΦX_sample = np.load('Data/X_sample.npy'), np.load('Data/Y_sample.npy'), np.load('Data/phi_x_sample.npy')
        β_sample = normal_formula(ΦX_sample, Y_sample)

        dots_sample = VGroup(*[Dot(axes.c2p(x, y), color=TEAL) for x, y in zip(X_sample.ravel(), Y_sample.ravel())])

        best_fit_graph_sample = axes.get_graph(lambda x: β_sample[0, 0] + x * β_sample[1, 0], color=RED_C)

        plot_sample_fit = Group(axes, x_label, y_label, dots_sample, best_fit_graph_sample)

        # En la realidad el grupo de personas...

        # tendríamos la siguiente gráfica
        X_population, Y_population = np.load('Data/X_population.npy'), np.load('Data/Y_population.npy')
        dots_population = VGroup(*[Dot(axes.c2p(x, y), color=TEAL) for x, y in zip(X_population.ravel(), Y_population.ravel())])
        for dot in dots_population:
            dot.scale(0.6)

        # si se calcula el promedio de cada grupo
        X_mean, Y_mean = np.load('Data/X_mean.npy'), np.load('Data/Y_mean.npy')
        ΦX_mean = np.hstack((np.ones((len(X_mean), 1)), X_mean))
        dots_mean = VGroup(*[Dot(axes.c2p(x, y), color=ORANGE) for x, y in zip(X_mean.ravel(), Y_mean.ravel())])

        # A este promedio por grupos se le llama...
        cond_exp = Tex(r'E(y_i|x_i)', color=ORANGE)
        # cond_exp.scale(0.9).to_corner(UR)
        cond_exp.scale(0.9).move_to(axes.c2p(40, 160))

        legends = create_legends()
        legends.move_to(RIGHT*5.3+UP)
        legends[1].scale(0.7)
        legends[3].scale(0.7)

        # Se puede trazar una línea que pase por cada promedio
        β_population = normal_formula(ΦX_mean, Y_mean)
        best_fit_graph_population = axes.get_graph(lambda x: β_population[0, 0] + x*β_population[1, 0], color=YELLOW_C)

        # en base a un fragmento o muestra de la población
                  # plot_sample_fit.animate.move_to(ORIGIN))

        # notemos que \hat{\beta}_2 es la pendiente
        x_coord = 200
        y_coord = float(np.array([[1, x_coord]])@β_sample)
        dot_x = Dot(axes.c2p(x_coord, β_sample[0,0]))
        dot_y = Dot(axes.c2p(x_coord, y_coord))

        v_line = DashedLine(dot_x.get_top()-[0, 0.08, 0], dot_y.get_bottom(), color=BLUE_C, stroke_width=3)
        h_line = axes.get_h_line(dot_x.get_left() + [0.08, 0, 0], color=BLUE_C, stroke_width=3)
        text_Δx = Tex(rf'\scriptstyle \Delta x={x_coord}', color=BLUE_C)
        text_Δx.next_to(h_line, UP)
        text_Δy = Tex(rf'\scriptstyle \Delta y={y_coord-β_sample[0,0]:.2f}', color=BLUE_C)
        text_Δy.next_to(v_line, RIGHT)

        beta_2 = Tex(rf'\hat{{\beta}}_2=\frac{{{y_coord-β_sample[0,0]:.2f}}}{{{x_coord}}}={β_sample[1,0]:.2f}',
                     color=BLUE_C)
        beta_2.scale(0.8).to_corner(UR)

        # Como cada predicción es el valor esperado de y dado x
        pred1 = Tex(r'\hat{y}_i', '=', r'\hat{\beta}_1', '+', r'\hat{\beta}_2', 'x_i', color=YELLOW_C)
        pred1.next_to(beta_2, DOWN, buff=MED_LARGE_BUFF)

        # Cuando x = 0
        dot_pred = Dot(axes.c2p(0, 0), color=YELLOW_C)
        pred2 = Tex(r'\hat{y}_i', '=', r'\hat{\beta}_1', '+', r'\hat{\beta}_2', '0', color=YELLOW_C)
        pred2.next_to(beta_2, DOWN, buff=MED_LARGE_BUFF)

        # la estimación toma el valor de \hat{\beta}_1 (beta 1 sombrero)
        pred3 = Tex(r'\hat{y}_i', '=', r'\hat{\beta}_1', color=YELLOW_C)
        pred3.next_to(beta_2, DOWN, buff=MED_LARGE_BUFF)

        print('Listo')
        wait_for_input(False)
        self.play(FadeIn(plot_sample_fit))
        wait_for_input()
        self.play(FadeOut(best_fit_graph_sample))
        wait_for_input()
        self.play(TransformFromCopy(dots_sample, dots_population), FadeOut(dots_sample))
        wait_for_input()
        self.play(FadeIn(dots_mean))
        wait_for_input()
        self.play(Write(cond_exp))
        wait_for_input()
        self.play(ShowCreation(best_fit_graph_population), FadeIn(legends[0]), FadeIn(legends[1]))
        wait_for_input()
        self.play(ShowCreation(best_fit_graph_sample),
                  FadeIn(legends[2]), FadeIn(legends[3]))
        wait_for_input()
        self.play(TransformFromCopy(dots_population, dots_sample), FadeOut(dots_population), FadeOut(legends),
                  FadeOut(best_fit_graph_population), FadeOut(dots_mean), FadeOut(cond_exp))
        wait_for_input()
        self.play(ShowCreation(v_line), ShowCreation(h_line), Write(text_Δy), Write(text_Δx))
        wait_for_input()
        self.play(Write(beta_2))
        wait_for_input()
        self.play(Write(pred1))
        wait_for_input()
        self.play(TransformMatchingTex(pred1, pred2), ShowCreation(dot_pred))
        wait_for_input()
        self.play(TransformMatchingTex(pred2, pred3),
                  dot_pred.animate.move_to(axes.c2p(0, β_sample[0, 0])))
        wait_for_input()


def create_planes(axes3d):
    plane1 = ParametricSurface(
        lambda u, v: np.array([
            axes3d.c2p(100, u, v)
        ]),
        u_range=(0, 200), v_range=(0, 30),
        opacity=0.5,
        color=GOLD_C,
        resolution=(10, 10))
    mesh1 = SurfaceMesh(plane1)
    mesh1.set_stroke(WHITE, 1, opacity=0.5)
    plane2 = ParametricSurface(
        lambda u, v: np.array([
            axes3d.c2p(u, v, 6)
        ]),
        u_range=(0, 300), v_range=(0, 200),
        opacity=0.5,
        color=GOLD_A,
        resolution=(10, 10))
    mesh2 = SurfaceMesh(plane2)
    mesh2.set_stroke(WHITE, 1, opacity=0.5)

    planes = Group(plane1, mesh1, plane2, mesh2)
    return planes


def plane_split(plane_points, points, counts):
    up = []
    down = []

    j = 0
    k = 0
    for _, v in counts.items():
        for i in range(v):
            if points[k+i].get_center()[1] < plane_points[j].get_center()[1]:
                down.append(points[k+i])
            else:
                up.append(points[k+i])
        k += v
        j += 1

    return Group(*up), Group(*down)


def plane_split_sample(plane_y, points_y, dots):
    up = []
    down = []

    for i, (y, p, dot) in enumerate(zip(plane_y.ravel(), points_y.ravel(), dots)):
        if p < y:
            down.append(dot)
        else:
            up.append(dot)
    return Group(*up), Group(*down)


def get_y(p):
    return p.get_center()[1]


class Multiple(ThreeDScene):
    def construct(self):
        delay = 0.5
        vector_line1 = Tex(r'\begin{bmatrix}\hat{y}_1\\\hat{y}_2\\\vdots\\\hat{y}_m\end{bmatrix}', r'=',
                           r'\begin{bmatrix}1,\hspace{2pt}x_{1,2},\hspace{2pt}x_{1,3}\\' +
                           r'1,\hspace{2pt}x_{2,2},\hspace{2pt}x_{2,3}\\' +
                           r'\vdots\\' +
                           r'1,\hspace{2pt}x_{m,2},\hspace{2pt}x_{m,3}\end{bmatrix}' +
                           r'\begin{bmatrix}\hat{\beta}_1\\\hat{\beta}_2\\\hat{\beta}_3\end{bmatrix}')
        # print('Listo')
        # wait_for_input(False)
        # wait_for_input()
        self.wait(11+delay)
        # Volviendo al modelo
        model1 = Tex(r'y_i', '=', r'\beta_1', '+', r'\beta_2', 'x_i')
        self.play(Write(model1))
        # wait_for_input()
        self.wait(2+delay)

        # Consideremos ahora una variable explicativa más
        model2 = Tex(r'y_i', '=', r'\beta_1', '+', r'\beta_2', 'x_i', '+', r'\beta_3', 'z_i')
        self.play(TransformMatchingTex(model1, model2))
        # wait_for_input()
        self.wait(7.8+delay)

        # Reescribiendo como
        model3 = Tex(r'y_i', '=', r'\beta_1', '+', r'\beta_2', 'x_{i2}', '+', r'\beta_3', 'x_{i3}')
        self.play(TransformMatchingTex(model2, model3))
        # wait_for_input()
        self.wait(10.3+delay)

        # Ahora contiene una nueva variable
        self.play(model3.animate.set_color_by_tex_to_color_map({'x_{i3}': YELLOW_C}))
        # wait_for_input()
        self.wait(1.3+delay)
        self.play(model3.animate.set_color_by_tex_to_color_map({r'\beta_3': BLUE_C}))
        # wait_for_input()
        self.wait(1.7+delay)

        # Entonces la esperanza condicional
        model4 = Tex(r'E(y_i|x_{i2},x_{i3})', '=', r'\beta_1', '+', r'\beta_2', 'x_{i2}', '+', r'\beta_3', 'x_{i3}')
        model4.fix_in_frame()
        self.play(TransformMatchingTex(model3, model4))
        # wait_for_input()
        self.wait(4.2+delay)

        # Si antes todos los valores de y_i...
        axes3d = ThreeDAxes(x_range=[0, 300, 40], y_range=[0, 200, 40], z_range=[0, 30, 5],
                            height=4, width=4, depth=4,
                            axis_config={'stroke_color': GREY_A, 'stroke_width': 2})
        axes3d.move_to(UP*0.3)
        # DESHACER ROTACIÓN
        angle_axes = 8 * DEGREES
        axes3d.rotate(angle_axes, np.array([0, 1, 0]))

        frame = self.camera.frame
        phi_frame = 6 * DEGREES
        frame.set_euler_angles(phi=phi_frame)

        x_label = axes3d.get_x_axis_label(r"\text{Ingreso semanal}", direction=DOWN)
        y_label = axes3d.get_y_axis_label(r"\text{Gasto de consumo semanal}", direction=RIGHT, buff=0.5)
        z_label = axes3d.get_axis_label(r'\text{Años de estudio}', axes3d.get_z_axis(), edge=UP, direction=LEFT,
                                        buff=SMALL_BUFF)
        x_label.scale(0.6).shift(LEFT*2+DOWN*0.3)
        y_label.scale(0.6).rotate(PI/2).shift(DOWN*2+LEFT*4)
        z_label.scale(0.6)

        plot = Group(axes3d, x_label, y_label, z_label)
        self.play(model4.animate.to_corner(UR),
                  FadeIn(axes3d), FadeIn(x_label), FadeIn(y_label))
        # wait_for_input()
        self.wait(1.6+delay)

        line = Line(axes3d.c2p(100, 0, 0), axes3d.c2p(100, 170, 0), color=GOLD_C)
        x_i = Tex('x_i', color=GOLD_C)
        x_i.scale(0.8).next_to(line, DOWN, buff=MED_SMALL_BUFF)

        dots_line = VGroup(*([Dot(axes3d.c2p(100, y, 0), color=TEAL_C) for y in [55, 60, 65, 70, 75]]))
        for dot in dots_line:
            dot.scale(0.6)
        self.play(FadeOut(model4), FadeIn(line), FadeIn(dots_line), FadeIn(x_i))
        # wait_for_input()
        self.wait(3.6+delay)

        # ahora el grupo está dado por todos los valores y_i en los planos...
        # self.play(FadeOut(dots_line), FadeOut(line), FadeOut(x_i),)
        self.play(FadeOut(dots_line), FadeOut(line), FadeOut(x_i),
                  FadeIn(z_label),
                  frame.animate.set_euler_angles(phi=-20*DEGREES),
                  plot.animate.rotate(-angle_axes-20*DEGREES, np.array([0, 1, 0])), run_time=1.5)
        # wait_for_input()
        self.wait(1.2+delay)

        X_population, Y_population = np.load('Data/X_mult_population.npy'), np.load('Data/Y_population.npy')
        X_mean, Y_mean = np.load('Data/X_mult_mean.npy'), np.load('Data/Y_mean.npy')
        ΦX_mean = np.hstack((np.ones((len(X_mean), 1)), X_mean))
        β_population = normal_formula(ΦX_mean, Y_mean)

        dots_population = VGroup(*[Dot(axes3d.c2p(x[0], y, x[1]), color=TEAL)
                                   for x, y in zip(X_population, Y_population.ravel())])
        for dot in dots_population:
            dot.scale(0.6)
        dots_population_slice = dots_population[:5].deepcopy()
        self.play(FadeIn(dots_population[:5]))
        # wait_for_input()
        self.wait(1+delay)

        planes = create_planes(axes3d)
        self.play(FadeIn(planes), FadeIn(dots_population_slice))
        # wait_for_input()
        self.wait(5+delay)

        graph_population = Group(plot, dots_population)

        # Al tener dos variables independientes y una dependiente, la gráfica ahora se debe realizar en 3 dimensiones.

        # Repitiendo el proceso de graficar la población
        self.play(FadeOut(planes),
                  FadeOut(dots_population_slice),
                  FadeIn(dots_population[5:]))
        # wait_for_input()
        self.wait(6.6+delay)

        # Graficando las esperanzas condicionales para cada grupo
        dots_mean = VGroup(*[Dot(axes3d.c2p(x[0], y, x[1]), color=ORANGE) for x, y in zip(X_mean, Y_mean.ravel())])
        graph_mean = Group(plot, dots_population, dots_mean)
        self.play(FadeIn(dots_mean))
        # wait_for_input()
        self.wait(3+delay)
        self.play(Rotate(graph_mean, 55*DEGREES, np.array([0, 1, 0])))
        # wait_for_input()
        # self.wait(0)

        # Particionando el plano
        counts = Counter(X_population[:,0].ravel())
        dots_population_up, dots_population_down = plane_split(dots_mean, dots_population, counts)

        population_fit = ParametricSurface(
            lambda u, v: np.array([
                axes3d.c2p(u, β_population[0, 0] + u*β_population[1, 0] + v*β_population[2, 0], v)
            ]),
            u_range=(0, 300), v_range=(0, 30),
            opacity=0.7,
            color=YELLOW_D,
            resolution=(10, 10))
        mesh_population = SurfaceMesh(population_fit)
        mesh_population.set_stroke(WHITE, 1, opacity=1)

        # Copia de los puntos arriba para volverlos a graficar
        dots_mean_copy = dots_mean.deepcopy()
        dots_population_up_copy = dots_population_up.deepcopy()
        self.play(ShowCreation(mesh_population), ShowCreation(population_fit),
                  FadeIn(dots_population_up_copy),
                  FadeIn(dots_mean_copy))
        # wait_for_input()
        self.wait(9+delay)

        full_plot = Group(plot, dots_population, dots_mean, mesh_population, population_fit,
                          dots_population_up_copy, dots_mean_copy)
        self.play(full_plot.animate.to_corner(LEFT))
        # wait_for_input()
        self.wait(0.7+delay)

        # De manera similar al caso con una variable
        line_eq1 = Tex(r'\hat{y}_i', r'=', r'\hat{\beta}_1 + \hat{\beta}_2x_{i2} + \hat{\beta}_3x_{i3}')
        line_eq1.move_to(RIGHT*3.5).fix_in_frame()
        self.play(Write(line_eq1))
        # wait_for_input()
        self.wait(1.2+delay)

        # vector_line1 = Tex(r'\begin{bmatrix}\hat{y}_1\\\hat{y}_2\\\vdots\\\hat{y}_m\end{bmatrix}', r'=',
        #                    r'\begin{bmatrix}1,\hspace{2pt}x_{1,2},\hspace{2pt}x_{1,3}\\'+
        #                    r'1,\hspace{2pt}x_{2,2},\hspace{2pt}x_{2,3}\\'+
        #                    r'\vdots\\'+
        #                    r'1,\hspace{2pt}x_{m,2},\hspace{2pt}x_{m,3}\end{bmatrix}'+
        #                    r'\begin{bmatrix}\hat{\beta}_1\\\hat{\beta}_2\\\hat{\beta}_3\end{bmatrix}')
        vector_line1.next_to(line_eq1, UP).scale(0.9).fix_in_frame()
        self.play(TransformMatchingTex(line_eq1, vector_line1))
        # wait_for_input()
        self.wait(1.9+delay)

        vector_line2 = Tex(r'\mathbf{\hat{y}} = X\hat{\beta}')
        vector_line2.next_to(vector_line1, DOWN, buff=MED_LARGE_BUFF).fix_in_frame()
        self.play(Write(vector_line2))
        # wait_for_input()
        self.wait(1.3+delay)

        normal_eq = Tex(r'\hat{\beta}', '=', r'(', r'\mathbf{X}^T\mathbf{X}', ')^{-1}', r'\mathbf{X}^T\mathbf{y}')
        normal_eq.next_to(vector_line2, DOWN, buff=MED_LARGE_BUFF).shift(UP).fix_in_frame()
        self.play(FadeOut(vector_line1), vector_line2.animate.shift(UP), Write(normal_eq))
        # wait_for_input()
        self.wait(9.6+delay)

        # Si se considera la muestra del conjunto de puntos, se tiene el plano ajustado:
        X_sample, Y_sample, ΦX_sample = np.load('Data/X_mult_sample.npy'), np.load('Data/Y_sample.npy'), np.load(
            'Data/phi_x_mult_sample.npy')
        β_sample = normal_formula(ΦX_sample, Y_sample)
        dots_sample = VGroup(*[Dot(axes3d.c2p(x[0], y, x[1]), color=TEAL) for x, y in zip(X_sample, Y_sample.ravel())])

        self.play(FadeOut(dots_population_up_copy),
                  FadeOut(dots_mean), FadeOut(mesh_population), FadeOut(population_fit), FadeOut(dots_mean_copy),
                  Transform(dots_population, dots_sample))
        # wait_for_input()
        self.wait(0.8+delay)

        plot_sample = Group(plot, dots_sample)
        self.remove(dots_population)
        self.play(plot_sample.animate.rotate(50 * DEGREES, np.array([0, 1, 0])))
        self.wait(3+delay)

        sample_fit = ParametricSurface(
            lambda u, v: np.array([
                axes3d.c2p(u, β_sample[0, 0] + u * β_sample[1, 0] + v * β_sample[2, 0], v)
            ]),
            u_range=(0, 300), v_range=(0, 30),
            opacity=0.7,
            color=RED_C,
            resolution=(10, 10))
        mesh_sample = SurfaceMesh(sample_fit)
        mesh_sample.set_stroke(WHITE, 1, opacity=1)

        dots_sample_up, dots_sample_down = plane_split_sample(ΦX_sample @ β_sample, Y_sample, dots_sample)
        dots_sample_up_copy = dots_sample_up.deepcopy()

        self.play(ShowCreation(mesh_sample), ShowCreation(sample_fit), FadeIn(dots_sample_up_copy))
        # wait_for_input()
        self.wait(5.2+delay)

        final_eq = Tex(r'\hat{y} = ',
                       r'\begin{bmatrix}1,\hspace{2pt}a_1,\hspace{2pt}\dots\hspace{2pt},a_{n-1}\end{bmatrix}',
                       r'\begin{bmatrix}\hat{\beta}_1\\\vdots\\\hat{\beta}_n\end{bmatrix}')
        final_eq.next_to(normal_eq, DOWN).fix_in_frame()
        self.play(Write(final_eq[0]))
        # wait_for_input()
        self.wait(4.3+delay)
        self.play(Write(final_eq[1]))
        # wait_for_input()
        self.wait(5.4+delay)
        self.play(Write(final_eq[2]))
        # wait_for_input()
        self.wait(2+delay)
