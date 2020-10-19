#version 430
layout(location = 0) uniform mat4 PVM;

layout(location = 0) in vec4 pos_attrib;
layout(location = 1) in vec4 color_attrib;

out vec4 color; //RGBindex

void main(void)
{
	gl_Position = PVM * pos_attrib;
	color = color_attrib;
}