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

// timer
float time;

const unsigned int tex_width = 512;	// tex_width of the figure
const unsigned int tex_height = 512;	// tex_height of the figure

// VBO vector field preview
const unsigned int vf_view_scale = 8;
const float vf_view_step = 1.0f / vf_view_scale;
const unsigned int vf_view_size = vf_view_scale * vf_view_scale * vf_view_scale;

// Vector field view
GLuint vbo = -1;
static const std::string vertex_shader("shader_vert.glsl");
static const std::string fragment_shader("shader_frag.glsl");
GLuint shader_program = -1;
struct cudaGraphicsResource* cuda_vbo_resource;
float3* cuda_vbo_result;

// 3D texture rendering
// diffusion result
GLuint diffusion_pbo = -1;
struct cudaGraphicsResource* cuda_diffusion_resource;
float3* cuda_diffusion_result;	// result of ray marching at diffusion
GLuint diffusion_pbo_texture = -1;	// texture for display
// velocity display
GLuint velocity_pbo = -1;
struct cudaGraphicsResource* cuda_velocity_resource;
float3* cuda_velocity_result;	// result of ray marching at velocity
GLuint velocity_pbo_texture = -1;	// texture for display

// Field data float4(Vx, Vy, Vz, P) velocity and pressure/power/energy
const unsigned int VF_data_scale = 256;
const unsigned int VF_data_size = VF_data_scale * VF_data_scale * VF_data_scale;
//float4* h_user_VF_inout = new float4[VF_data_size];	// User input
// two VF buffer for swap
float4* d_VF_0;
float4* d_VF_1;
//float4* d_user_input_VF;
bool useVF_0 = true;	// ping-pong buffer flag
float3* d_gradient;
float* d_divergence;

// camera
float3 viewRotation = make_float3(0.0f);
float3 viewTranslation = make_float3(0.0, 0.0, -5.0f);
float invViewMatrix[12];

// init
extern "C" void launch_init_VF_kernel(float4* VF);
extern "C" void createTransferTexture();

// calculate
extern "C" void launch_gradient_kernel(float4* VF, float3* gradient);
extern "C" void launch_divergence_kernel(float3* gradient, float* divergence);
extern "C" void launch_update_VF_kernel(float4* pre_VF, float4* current_VF, float* divergence);

// display
extern "C" void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix);
extern "C" void launch_vbo_kernel(float3* cuda_vbo_result, float4* VF, unsigned int vf_view_scale);
extern "C" void launch_display_kernel(float4* VF, float3* gradient, float* divergence,
	float3* cuda_diffusion_result,
	float3* cuda_velocity_result,
	float density, float transferOffset, float transferScale);

// kernelFunc
extern "C" void checkCudaError(const char* msg);
extern "C" void freeCudaTextureBuffers();


// imgui
// ray marching parameters
float density = 0.04f;
float transferOffset = 0.0f;
float transferScale = 1.0f;


void resetCamera()
{
	viewRotation = make_float3(0.0f);
	viewTranslation = make_float3(0.0, 0.0, -5.0f);
}

void draw_gui()
{
	ImGui_ImplGlut_NewFrame();
	//ImGui::ShowDemoWindow();
	ImGui::Begin("Main control");
	if (ImGui::Button("Reset Camera"))
	{
		resetCamera();
	}

	ImGui::PushItemWidth(240);

	ImGui::SliderFloat("Volume Density", &density, 0.0f, 0.2f);
	ImGui::SliderFloat("Volume Transfer Offset", &transferOffset, -1.0f, 1.0f);
	ImGui::SliderFloat("Volume Transfer Scale", &transferScale, 0.0f, 2.0f);
	ImGui::End();

	ImGui::Begin("Diffusion");
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, diffusion_pbo);
	glBindTexture(GL_TEXTURE_2D, diffusion_pbo_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex_width, tex_height,
		GL_RGB, GL_FLOAT, NULL);
	ImGui::Image((void*)diffusion_pbo_texture, ImVec2(512, 512));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	ImGui::End();

	ImGui::Begin("Gradient");
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, velocity_pbo);
	glBindTexture(GL_TEXTURE_2D, velocity_pbo_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex_width, tex_height,
		GL_RGB, GL_FLOAT, NULL);
	ImGui::Image((void*)velocity_pbo_texture, ImVec2(512, 512));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	ImGui::End();

	ImGui::Render();
}

void createVBO()
{
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * vf_view_size * 4, 0, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

// create pixel buffer objects
void createPBOs()
{
	int num_texels = tex_width * tex_height;
	int num_values = num_texels * 3;

	int size_tex_data = sizeof(GLfloat) * num_values;

	// diffusion
	glGenBuffers(1, &diffusion_pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, diffusion_pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(&cuda_diffusion_resource, diffusion_pbo, cudaGraphicsMapFlagsWriteDiscard);

	// velocity
	glGenBuffers(1, &velocity_pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, velocity_pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGraphicsGLRegisterBuffer(&cuda_velocity_resource, velocity_pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void createTextures()
{
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &diffusion_pbo_texture);
	glBindTexture(GL_TEXTURE_2D, diffusion_pbo_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, tex_width, tex_height, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &velocity_pbo_texture);
	glBindTexture(GL_TEXTURE_2D, velocity_pbo_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, tex_width, tex_height, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void initCuda()
{
	cudaMalloc((void**)& d_VF_0, sizeof(float4) * VF_data_size);
	checkCudaError("Cuda Malloc d_VF_0 failed!");
	cudaMalloc((void**)& d_VF_1, sizeof(float4) * VF_data_size);
	checkCudaError("Cuda Malloc d_VF_1 failed!");

	cudaMalloc((void**)& d_gradient, sizeof(float3) * VF_data_size);
	checkCudaError("Cuda Malloc d_gradient failed!");
	cudaMalloc((void**)& d_divergence, sizeof(float) * VF_data_size);
	checkCudaError("Cuda Malloc d_divergence failed!");

	// fill d_VF_0
	launch_init_VF_kernel(d_VF_0);

	createVBO();
	createPBOs();
	createTextures();

	size_t num_bytes;
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)& cuda_vbo_result, &num_bytes, cuda_vbo_resource);

	cudaGraphicsMapResources(1, &cuda_diffusion_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)& cuda_diffusion_result, &num_bytes, cuda_diffusion_resource);
	cudaMemset(cuda_diffusion_result, 0, tex_width * tex_height * 3);

	cudaGraphicsMapResources(1, &cuda_velocity_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)& cuda_velocity_result, &num_bytes, cuda_velocity_resource);
	cudaMemset(cuda_velocity_result, 0, tex_width * tex_height * 3);
}

void runCuda()
{
	if (useVF_0)	// using VF_0 as input = use volume_0 as input
	{
		// calculate
		launch_gradient_kernel(d_VF_0, d_gradient);
		launch_divergence_kernel(d_gradient, d_divergence);
		launch_update_VF_kernel(d_VF_0, d_VF_1, d_divergence);
		// display
		launch_vbo_kernel(cuda_vbo_result, d_VF_1, vf_view_scale);
		launch_display_kernel(d_VF_1, d_gradient, d_divergence, 
			cuda_diffusion_result,
			cuda_velocity_result,
			density, transferOffset, transferScale);
	}
	else
	{
		// calculate
		launch_gradient_kernel(d_VF_1, d_gradient);
		launch_divergence_kernel(d_gradient, d_divergence);
		launch_update_VF_kernel(d_VF_1, d_VF_0, d_divergence);
		// display
		launch_vbo_kernel(cuda_vbo_result, d_VF_0, vf_view_scale);
		launch_display_kernel(d_VF_0, d_gradient, d_divergence,
			cuda_diffusion_result,
			cuda_velocity_result,
			density, transferOffset, transferScale);
	}

	useVF_0 = !useVF_0;	// swap
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
	glRotatef(viewRotation.x, 1.0, 0.0, 0.0);
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
	
	const int w = glutGet(GLUT_WINDOW_WIDTH);
	const int h = glutGet(GLUT_WINDOW_HEIGHT);
	const float aspect_ratio = float(w) / float(h);

	// offset
	glm::mat4 M = glm::translate(glm::vec3(vf_view_step));

	const int PVM_loc = 0;

	float viewMat[16];

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glTranslatef(viewTranslation.x, viewTranslation.y, viewTranslation.z);
	glRotatef(viewRotation.y, 0.0, 1.0, 0.0);
	glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
	glGetFloatv(GL_MODELVIEW_MATRIX, viewMat);
	glPopMatrix();

	glm::mat4 V = glm::mat4(viewMat[0], viewMat[1], viewMat[2], viewMat[3],
							viewMat[4], viewMat[5], viewMat[6], viewMat[7],
							viewMat[8], viewMat[9], viewMat[10], viewMat[11],
							viewMat[12], viewMat[13], viewMat[14], viewMat[15]);

	glm::mat4 P = glm::perspective(3.141592f / 4.0f, aspect_ratio, 0.1f, 100.0f);

	glm::mat4 PVM = P * V * M;

	glUseProgram(shader_program);
	glUniformMatrix4fv(PVM_loc, 1, false, glm::value_ptr(PVM));

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glEnableClientState(GL_VERTEX_ARRAY);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));

	glLineWidth(1.5f);
	glDrawArrays(GL_LINES, 0, vf_view_size *4);
	//glPointSize(2.0f);
	//glDrawArrays(GL_POINTS, 0, dataSize*4);

	glDisableClientState(GL_VERTEX_ARRAY);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);	// in case for imgui's bug
}

void idle()
{
	glutPostRedisplay();

	updateCamera();

	// timer
	//time = glutGet(GLUT_ELAPSED_TIME)*0.001f;
	runCuda();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	const int w = glutGet(GLUT_WINDOW_WIDTH);
	const int h = glutGet(GLUT_WINDOW_HEIGHT);
	const float aspect_ratio = float(w) / float(h);

	drawVertexField();
	
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

void SetOpenGLDevice()
{
	cudaDeviceProp prop;
	int dev;
	//fill it with zeros memset(&prop,0,sizeof(cudaDeviceProp)); 
	prop.major = 1; prop.minor = 0;
	//pick a GPU capable of 1.0 or better
	cudaChooseDevice(&dev, &prop);
	cudaError_t error = cudaGLSetGLDevice(dev); //set OpenGL device
	if (error != cudaSuccess)
	{
		std::cout << "Set device failed!" << std::endl;
		return;
	}
}


void initOpenGl()
{
	SetOpenGLDevice();
	//Initialize glew so that new OpenGL function names can be used
	glewInit();

	glViewport(0, 0, tex_width, tex_height);
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
	if (!ImGui::IsMouseHoveringAnyWindow())
	{
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
		viewTranslation.y += dy / 100.0f;
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
	glutInitWindowSize(1920, 1080);
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
	initCuda();
	createTransferTexture();

	//Enter the glut event loop.
	glutMainLoop();
	cudaThreadExit();
	glutDestroyWindow(win);

	// free buffer before close
	freeCudaTextureBuffers();
	cudaFree(cuda_diffusion_result);
	cudaFree(cuda_velocity_result);
	cudaFree(d_VF_0);
	cudaFree(d_VF_1);
	cudaFree(d_gradient);
	cudaFree(d_divergence);

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	cudaGraphicsUnmapResources(1, &cuda_diffusion_resource, 0);
	cudaGraphicsUnmapResources(1, &cuda_velocity_resource, 0);

	ImGui_ImplGlut_Shutdown();

	return 0;
}