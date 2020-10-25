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

// timer
float time;

// 3D texture rendering
GLuint pbo = -1;
struct cudaGraphicsResource* cuda_pbo_resource;	// pointer to the returned object handle
float3* cuda_pbo_result;		// place for CUDA output
// 3D texture data
const char* volumeFileName = "data/Bucky.raw";
cudaExtent volumeScale = make_cudaExtent(32, 32, 32);
typedef unsigned char VolumeType;
float density = 0.04f;
float transferOffset = 0.0f;
float transferScale = 1.0f;

// Vector field
GLuint vao = -1;
GLuint vbo = -1;
GLuint ebo = -1;
static const std::string vertex_shader("shader_vert.glsl");
static const std::string fragment_shader("shader_frag.glsl");
GLuint shader_program = -1;

struct cudaGraphicsResource* cuda_vbo_resource;
float3* cuda_vbo_result;

float3* h_VF_input = new float3[dataSize];	// User input vectors
float3* d_VF_input;
unsigned int currentPickedIndex = 0;
int currentPickedX = 0;
int currentPickedY = 0;
int currentPickedZ = 0;
float currentVect[3] = { 0.0f,0.0f,0.0f };
float3 previewVect;

// camera
float3 viewRotation = make_float3(0.0f);
float3 viewTranslation = make_float3(0.0, 0.0, -5.0f);
float invViewMatrix[12];

// Implement of this function is in kernel.cu
extern "C" void launch_pbo_kernel(float3* cuda_pbo_result, unsigned int width, unsigned int height,
	float density, float transferOffset, float transferScale, float3* d_VF_input, unsigned int N);
extern "C" void launch_vbo_kernel(float3* cuda_vbo_result, unsigned int vf_scale, unsigned int currentPickedIndex, 
	float3* d_VF_input, float3 previewVect, float time);
extern "C" void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix);
extern "C" void copyVolumeTextures(void* h_volume, cudaExtent volumeScale);
extern "C" void checkCudaError(const char* msg);


// imgui
GLuint pbo_texture = -1;


void resetCamera()
{
	viewRotation = make_float3(0.0f);
	viewTranslation = make_float3(0.0, 0.0, -5.0f);
}

void draw_gui()
{
	ImGui_ImplGlut_NewFrame();
	//ImGui::ShowDemoWindow();
	ImGui::Begin("Settings");
	if(ImGui::Button("Reset Camera"))
	{
		resetCamera();
	}
	ImGui::PushItemWidth(240);
	ImGui::SliderInt("Pick Vector X", &currentPickedX, 0, vf_scale - 1);
	ImGui::SliderInt("Pick Vector Y", &currentPickedY, 0, vf_scale - 1);
	ImGui::SliderInt("Pick Vector Z", &currentPickedZ, 0, vf_scale - 1);
	currentPickedIndex = currentPickedX * vf_scale * vf_scale + currentPickedY * vf_scale + currentPickedZ;

	ImGui::SliderFloat3("Modify Vector", currentVect, -1.0f, 1.0f);
	previewVect = make_float3(currentVect[0], currentVect[1], currentVect[2]);
	if (ImGui::Button("Set Vector"))
	{
		h_VF_input[currentPickedIndex] = previewVect;
	}

	ImGui::SliderFloat("Volume Density", &density, 0.0f, 0.2f);
	ImGui::SliderFloat("Volume Transfer Offset", &transferOffset, -1.0f, 1.0f);
	ImGui::SliderFloat("Volume Transfer Scale", &transferScale, 0.0f, 2.0f);
	ImGui::End();

	ImGui::Begin("Ray marching");
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBindTexture(GL_TEXTURE_2D, pbo_texture);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
		GL_RGB, GL_FLOAT, NULL);
	ImGui::Image((void*)pbo_texture, ImVec2(512, 512));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	ImGui::End();

	ImGui::Render();
}

float uniformRand()	//(-1,1)
{
	return (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

void GenVectorField()
{
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * dataSize * 4, 0, GL_STATIC_DRAW);

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

void createTexture()
{
	glEnable(GL_TEXTURE_2D);

	glGenTextures(1, &pbo_texture);
	glBindTexture(GL_TEXTURE_2D, pbo_texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

}

void initInputVF()
{
	memset(h_VF_input, 0, sizeof(float3) * dataSize);
	//for (int i = 0; i < dataSize; i++)
	//{
	//	h_VF_input[i].x = sin(i / 4.0f);
	//	h_VF_input[i].y = cos(i / 4.0f);
	//	h_VF_input[i].z = sin(i / 4.0f);
	//}
}

void setBuffers()
{
	initInputVF();

	cudaMalloc((void**)& d_VF_input, sizeof(float3) * dataSize);
	checkCudaError("Cuda Malloc d_VF_input failed!");
	GenVectorField();
	createPBO();
	createTexture();
}

void setVolumeTextures()
{
	size_t volumeDataSize = sizeof(VolumeType) * volumeScale.width * volumeScale.height * volumeScale.depth;
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

	cudaMemcpy(d_VF_input, h_VF_input, sizeof(float3) * dataSize, cudaMemcpyHostToDevice);
	checkCudaError("Cuda Memcpy d_VF_input failed!");
	launch_vbo_kernel(cuda_vbo_result, vf_scale, currentPickedIndex, d_VF_input, previewVect, time);

	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

	// render 3D texture kernel
	
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)& cuda_pbo_result, &num_bytes, cuda_pbo_resource);

	cudaMemset(cuda_pbo_result, 0, width * height * 3);

	launch_pbo_kernel(cuda_pbo_result, width, height, density, transferOffset, transferScale, d_VF_input, vf_scale);

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
	glm::mat4 M = glm::translate(glm::vec3(vf_step));

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
	glDrawArrays(GL_LINES, 0, dataSize*4);
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
	time = glutGet(GLUT_ELAPSED_TIME)*0.001f;
	runCuda();
	
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
	cudaFree(d_VF_input);

	delete[] h_VF_input;

	ImGui_ImplGlut_Shutdown();

	return 0;
}