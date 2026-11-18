from manim import *


Вот очищенный код с лаконичными английскими комментариями в академическом стиле:

```python
from manim import *


class EvasionScene(Scene):

  def construct(self):
    # 1. Initialize geometry: Arch (exit boundary), Evader agent, Patrol inspector
    arch = Line(UP * 2, DOWN * 2, color=BLUE).move_to(LEFT * 4)
    evader = Dot(color=GREEN, radius=0.2).move_to(LEFT * 1)
    patrol = Dot(color=RED, radius=0.2).move_to(RIGHT * 3)

    # 2. Mathematical labels
    evader_label = MathTex(r"E_t").next_to(evader, UP)
    patrol_label = MathTex(r"P_t").next_to(patrol, UP)
    arch_label = Text("Safe Zone (Arch)", font_size=24).next_to(arch, LEFT)

    # --- ANIMATION SEQUENCE ---

    # Display initial elements
    self.play(Create(arch), Write(arch_label))
    self.play(FadeIn(evader, evader_label), FadeIn(patrol, patrol_label))
    self.wait(0.5)

    # Pursuit dynamics: evader moves to exit while patrol intercepts
    self.play(
        evader.animate.move_to(arch.get_center() + RIGHT * 0.2),
        evader_label.animate.move_to(arch.get_center() + RIGHT * 0.2 + UP),
        patrol.animate.move_to(LEFT * 1),
        patrol_label.animate.move_to(LEFT * 1 + UP),
        run_time=2,
        rate_func=rush_from,  # Physical acceleration curve
    )

    self.wait(2)

