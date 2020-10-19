#version 430
layout(location = 1) uniform int currentIndex;
in vec4 color;

out vec4 fragcolor;

void main(void)
{
	if (currentIndex - color.w < 0.0001f) fragcolor = vec4(1.0f, 1.0f, 0.0f, 1.0f);
	fragcolor = vec4(color.xyz, 1.0f);
}