from manim import *


class EvasionScene(Scene):
    def construct(self):
        # 1. Создаем объекты: Арка (синяя линия), Уклонист (зеленая точка), Патруль (красная)
        arch = Line(UP * 2, DOWN * 2, color=BLUE).move_to(LEFT * 4)
        evader = Dot(color=GREEN, radius=0.2).move_to(LEFT * 1)
        patrol = Dot(color=RED, radius=0.2).move_to(RIGHT * 3)

        # 2. Добавляем строгие математические подписи
        evader_label = MathTex(r"E_t").next_to(evader, UP)
        patrol_label = MathTex(r"P_t").next_to(patrol, UP)
        arch_label = Text("Safe Zone (Arch)", font_size=24).next_to(arch, LEFT)

        # --- ЗАПУСК АНИМАЦИИ ---

        # Появление объектов на экране
        self.play(Create(arch), Write(arch_label))
        self.play(FadeIn(evader, evader_label), FadeIn(patrol, patrol_label))
        self.wait(0.5)

        # Симуляция погони: уклонист бежит к арке, патруль двигается за ним
        self.play(
            evader.animate.move_to(arch.get_center() + RIGHT * 0.2),
            evader_label.animate.move_to(arch.get_center() + RIGHT * 0.2 + UP),
            patrol.animate.move_to(LEFT * 1),
            patrol_label.animate.move_to(LEFT * 1 + UP),
            run_time=2,
            rate_func=rush_from  # Имитация физического ускорения
        )

        self.wait(2)