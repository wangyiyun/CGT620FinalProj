// refer to CUDA example: volumeRender
#include <stdio.h>
#include <windows.h>
// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/gl.h>
// OpenGL math lib
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <fstream>
#include <string>

#include "cuda_gl_interop.h"

#include "cutil_math.h"
#include <curand.h>
#include <curand_kernel.h>

#include "include/FreeImage.h"
#include "imgui/imgui_impl_glut.h"
#include "InitShader.h"

#include "VectorField.h"

const int width = 512;	// width of the figure
const int height = 512;	// height of the figure
const int vf_scale = 8;
const float vf_step = 1.0f / vf_scale;
const int dataSize = vf_scale * vf_scale * vf_scale;

// 3D texture rendering
GLuint pbo = -1;	// pxiel buffer object, place for OpenGL and CUDA to switch data and display the result
GLuint textureID = -1;	// OpenGL texture to display the result
struct cudaGraphicsResource* cuda_pbo_resource;	// pointer to the returned object handle
float3* cuda_pbo_result;		// place for CUDA output
// 3D texture data
const char* volumeFileName = "data/Bucky.raw";
cudaExtent volumeScale = make_cudaExtent(32, 32, 32);
typedef unsigned char VolumeType;


// Vector field
GLuint vao = -1;
GLuint vbo = -1;
static const std::string vertex_shader("shader_vert.glsl");
static const std::string fragment_shader("shader_frag.glsl");
GLuint shader_program = -1;

struct cudaGraphicsResource* cuda_vbo_resource;
float3* cuda_vbo_result;

// camera
GLfloat camX, camZ;
GLfloat radius = 5.0f;
float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -5.0f);
float invViewMatrix[12];

// Implement of this function is in kernel.cu
extern "C" void launch_pbo_kernel(float3* cuda_pbo_result, unsigned int width, unsigned int height);
extern "C" void launch_vbo_kernel(float3* cuda_vbo_result, unsigned int vf_scale);
extern "C" void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix);
extern "C" void copyVolumeTextures(void* h_volume, cudaExtent volumeScale);


// imgui
bool renderVBO = true;

void draw_gui()
{
	ImGui_ImplGlut_NewFrame();
	//ImGui::ShowDemoWindow();
	ImGui::Begin("test");
	ImGui::Checkbox("render VBO", &renderVBO);
	ImGui::End();
	ImGui::Render();
}

float uniformRand()	//(-1,1)
{
	return (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

//void createVBO();
//void createVAO()
//{
//	glGenVertexArrays(1, &vao);
//	glBindVertexArray(vao);
//
//	createVBO();
//
//	const GLuint pos_loc = 0;
//	glEnableVertexAttribArray(pos_loc);
//	// position attribute
//	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
//	glEnableVertexAttribArray(0);
//
//	glBindVertexArray(0);
//}

void createVBO()
{
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * dataSize * 2, vfData, GL_DYNAMIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * dataSize * 2, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

// create pixel buffer object in OpenGL
void createPBO()
{
	int num_texels = width * height;
	int num_values = num_texels * 3;

	int size_tex_data = sizeof(GLfloat) * num_values;

	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);

	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

// create texture in OpenGL
void createTexture(GLuint* textureID, unsigned int size_x, unsigned int size_y)
{
	glEnable(GL_TEXTURE_2D);

	glGenTextures(1, textureID);
	glBindTexture(GL_TEXTURE_2D, *textureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void setBuffers()
{
	createVBO();
	createPBO();
	createTexture(&textureID, width, height);
}

void setVolumeTextures()
{
	size_t volumeDataSize = volumeScale.width * volumeScale.height * volumeScale.depth * sizeof(VolumeType);
	FILE* fp = fopen(volumeFileName, "rb");
	if (!fp)
	{
		fprintf(stderr, "Error opening file '%s'\n", volumeFileName);
		return;
	}
	void* h_volume = malloc(volumeDataSize);
	if (h_volume)
	{
		size_t read = fread(h_volume, 1, volumeDataSize, fp);
		printf("Read '%s', %zu bytes\n", volumeFileName, read);
	}
	else printf("Malloc '%s' failed!\n", volumeFileName);
	
	copyVolumeTextures(h_volume, volumeScale);
	
	free(h_volume);

	return;
}

void runCuda()
{
	size_t num_bytes;

	// update vector field kernel
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)& cuda_vbo_result, &num_bytes, cuda_vbo_resource);

	launch_vbo_kernel(cuda_vbo_result, vf_scale);

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

	// render 3D texture kernel
	
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)& cuda_pbo_result, &num_bytes, cuda_pbo_resource);

	cudaMemset(cuda_pbo_result, 0, width * height * 3);

	launch_pbo_kernel(cuda_pbo_result, width, height);

	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void display()
{

}


GLfloat modelView[16] =
{
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 4.0f, 1.0f
};

void updateCamera()
{
	// use OpenGL to build view matrix
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
	glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
	glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glPopMatrix();

	invViewMatrix[0] = modelView[0];
	invViewMatrix[1] = modelView[4];
	invViewMatrix[2] = modelView[8];
	invViewMatrix[3] = modelView[12];
	invViewMatrix[4] = modelView[1];
	invViewMatrix[5] = modelView[5];
	invViewMatrix[6] = modelView[9];
	invViewMatrix[7] = modelView[13];
	invViewMatrix[8] = modelView[2];
	invViewMatrix[9] = modelView[6];
	invViewMatrix[10] = modelView[10];
	invViewMatrix[11] = modelView[14];

	copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);
}

void drawVertexField()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(shader_program);
	const int w = glutGet(GLUT_WINDOW_WIDTH);
	const int h = glutGet(GLUT_WINDOW_HEIGHT);
	const float aspect_ratio = float(w) / float(h);

	//camX = sin(glutGet(GLUT_ELAPSED_TIME)*0.002f) * radius;
	//camZ = cos(glutGet(GLUT_ELAPSED_TIME)*0.002f) * radius;

	//glm::vec3 dir;
	//dir.x = cos(glm::radians(viewRotation.x))* cos(glm::radians(viewRotation.y));
	//dir.y = sin(glm::radians(viewRotation.y));
	//dir.x = sin(glm::radians(viewRotation.x)) * cos(glm::radians(viewRotation.y));
	//dir = glm::normalize(dir);

	glm::mat4 trans = glm::mat4(1.0f);
	glm::mat4 M = glm::rotate(trans, glm::radians(viewRotation.x), glm::vec3(-1.0f, 0.0f, 0.0f))
		* glm::rotate(trans, glm::radians(viewRotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 V = glm::lookAt(glm::vec3(0.0f, 0.0f, viewTranslation.z), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 P = glm::perspective(3.141592f / 4.0f, aspect_ratio, 0.1f, 100.0f);

	const int PVM_loc = 0;
	glm::mat4 PVM = P * V * M;
	glUniformMatrix4fv(PVM_loc, 1, false, glm::value_ptr(PVM));

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_LINES, 0, dataSize * 2);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);	// in case for imgui's bug
}

void drawPBO()
{
	glUseProgram(0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	glBindTexture(GL_TEXTURE_2D, textureID);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
		GL_RGB, GL_FLOAT, NULL);

	// draw a quadrangle as large as the window
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
	glEnd();
	// unbind
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void idle()
{
	glutPostRedisplay();

	updateCamera();

	runCuda();
	
	if(renderVBO) drawVertexField();
	else drawPBO();

	draw_gui();

	glutSwapBuffers();
}

// Display info about the OpenGL implementation provided by the graphics driver.
void printGlInfo()
{
	std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Version: " << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
	int X, Y, Z, total;
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &X);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &Y);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &Z);
	glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &total);
	std::cout << "Max Compute Work Group Size: " << X << " " << Y << " " << Z << std::endl;
	std::cout << "Max Compute Work Group Invocations: " << total << std::endl;
}

void reload_shader()
{
	GLuint new_shader = InitShader(vertex_shader.c_str(), fragment_shader.c_str());

	if (new_shader == -1) // loading failed
	{
		glClearColor(1.0f, 0.0f, 1.0f, 0.0f);
	}
	else
	{
		glClearColor(0.35f, 0.35f, 0.35f, 0.0f);

		if (shader_program != -1)
		{
			glDeleteProgram(shader_program);
		}
		shader_program = new_shader;

	}
}

void initOpenGl()
{
	//Initialize glew so that new OpenGL function names can be used
	glewInit();

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_DEPTH_TEST);

	reload_shader();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

// glut callbacks need to send keyboard and mouse events to imgui
void keyboard(unsigned char key, int x, int y)
{
	ImGui_ImplGlut_KeyCallback(key);
}
// some callback functions here
void keyboard_up(unsigned char key, int x, int y)
{
	ImGui_ImplGlut_KeyUpCallback(key);
}

void special_up(int key, int x, int y)
{
	ImGui_ImplGlut_SpecialUpCallback(key);
}

void passive(int x, int y)
{
	ImGui_ImplGlut_PassiveMouseMotionCallback(x, y);
}

void special(int key, int x, int y)
{
	ImGui_ImplGlut_SpecialCallback(key);
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
	ImGui_ImplGlut_MouseButtonCallback(button, state);
	//std::cout << x << " " << y << std::endl;
	if (state == GLUT_DOWN)
	{
		buttonState |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}

	ox = x;
	oy = y;
	glutPostRedisplay();

}

void motion(int x, int y)
{
	ImGui_ImplGlut_MouseMotionCallback(x, y);
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	if (buttonState == 4)
	{
		// right = zoom
		viewTranslation.z += dy / 100.0f;
	}
	else if (buttonState == 2)
	{
		// middle = translate
		viewTranslation.x += dx / 100.0f;
		viewTranslation.y -= dy / 100.0f;
	}
	else if (buttonState == 1)
	{
		// left = rotate
		viewRotation.x += dy / 5.0f;
		viewRotation.y += dx / 5.0f;
	}

	ox = x;
	oy = y;
	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	//Configure initial window state using freeglut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowPosition(5, 5);
	glutInitWindowSize(width, height);
	int win = glutCreateWindow("CGT620FinalProj");

	//Register callback functions with glut. 
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutKeyboardUpFunc(keyboard_up);
	glutSpecialUpFunc(special_up);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutPassiveMotionFunc(motion);

	glutIdleFunc(idle);

	initOpenGl();
	printGlInfo();
	// initialize the imgui system
	ImGui_ImplGlut_Init();	

	// buffers setup
	setBuffers();	// vbo, pbo
	setVolumeTextures();

	//Enter the glut event loop.
	glutMainLoop();
	cudaThreadExit();
	glutDestroyWindow(win);

	// free buffer before close
	cudaFree(cuda_pbo_result);
	cudaFree(cuda_vbo_result);

	ImGui_ImplGlut_Shutdown();

	return 0;
}