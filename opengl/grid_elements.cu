// from https://learnopengl.com/Getting-started/Hello-Triangle
#include <stdio.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#define WIDTH 640
#define HEIGHT 480
#define ROWS 10
#define COLS 10

#define N_CELLS ROWS * COLS
#define N_VERTEX (ROWS + 1) * (COLS + 1)

const char *vertexShaderSource = "#version 460 core\n"
                                 "layout (location = 0) in vec3 aPos;\n"
                                 "void main() {\n"
                                 "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
                                 "}\0";

const char *fragmentShaderSource = "#version 460 core\n"
                                   "out vec4 FragColor;\n"
                                   "void main() {\n"
                                   "    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
                                   "}\0";

unsigned int shaderProgram;
unsigned int VAO;

typedef struct _vec3 {
    float x, y, z;
    _vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {};
} vec3;


void setupShaderProgram() {
    // variables to store shader compiling errors
    int  success;
    char infoLog[512];

    // create our vertex shader
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    // compile the shader code
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check compiler errors
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
        exit(1);
    }

    // create our fragment shader
    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    // compile the shader code
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check compiler errors
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n", infoLog);
        exit(1);
    }

    // create a shader program to link our shaders
    shaderProgram = glCreateProgram();
    // link the shaders
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check linker errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        fprintf(stderr, "ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
        exit(1);
    }
    // clear already linked shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void display() {
    // use configured shaders
    glUseProgram(shaderProgram);
    // bind VAO, which implicitly binds our EBO
    glBindVertexArray(VAO);
    // params:
    //      mode / OpenGL primitive
    //      count of elements to draw (6 vertices)
    //      type of the indices
    //      offset in the EBO or an index array
    glDrawElements(GL_QUADS, 4 * N_CELLS, GL_UNSIGNED_INT, 0);
    // glBindVertexArray(0); // no need to unbind it every time 

    // actually in the screen
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
}

void initGL(int * argc, char ** argv) {
    // init glut
    glutInit(argc, argv);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("VBO Test");
    glutDisplayFunc(display);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    // init glew
    // this is required to setup GL functions (like glGenBuffers)
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        /* glewInit failed, something is seriously wrong */
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char **argv) {
    ///// INIT GL
    initGL(&argc, argv);

    ///// SHADER CONFIGURATION
    setupShaderProgram();
    

    ///// OBJECT & BUFFER CONFIGURATION

    // define square vertices
    // size_t verticesSize = N_VERTEX * sizeof(vec3);
    // vec3 * vertices = (vec3 *) malloc(verticesSize);
    // vertices[0] = vec3(-1.0f,  1.0f, 0.0f);  // 0 top left 
    // vertices[1] = vec3( 0.0f,  1.0f, 0.0f);  // 1 top center
    // vertices[2] = vec3( 1.0f,  1.0f, 0.0f);  // 2 top right
    // vertices[3] = vec3(-1.0f,  0.0f, 0.0f);  // 3 middle left
    // vertices[4] = vec3( 0.0f,  0.0f, 0.0f);  // 4 middle center
    // vertices[5] = vec3( 1.0f,  0.0f, 0.0f);  // 5 middle right
    // vertices[6] = vec3(-1.0f, -1.0f, 0.0f);  // 6 bottom left
    // vertices[7] = vec3( 0.0f, -1.0f, 0.0f);  // 7 bottom center
    // vertices[8] = vec3( 1.0f, -1.0f, 0.0f);  // 8 bottom right

    // size_t indicesSize = 4 * N_CELLS * sizeof(unsigned int);
    // unsigned int * indices = (unsigned int *) malloc(indicesSize);
    // int idx = 0;
    // indices[idx++] = 0;
    // indices[idx++] = 1;
    // indices[idx++] = 4;
    // indices[idx++] = 3;
    // indices[idx++] = 1;
    // indices[idx++] = 2;
    // indices[idx++] = 5;
    // indices[idx++] = 4;
    // indices[idx++] = 3;
    // indices[idx++] = 4;
    // indices[idx++] = 7;
    // indices[idx++] = 6;
    // indices[idx++] = 4;
    // indices[idx++] = 5;
    // indices[idx++] = 8;
    // indices[idx++] = 7;


    // since we're talking about vertices of squares, we have to add an extra row/col for the borders
    size_t verticesSize = N_VERTEX * sizeof(vec3);
    vec3 * vertices = (vec3 *) malloc(verticesSize);
    for (int y = 0; y < ROWS + 1; ++y) {
        for (int x = 0; x < COLS + 1; ++x) {
            int idx = y * (COLS + 1) + x;

            vertices[idx] = vec3(-1.0f + x * (2.0 / COLS), -1.0f + y * (2.0 / ROWS), 0.0f);
        }
    }
    // mount the squares
    size_t indicesSize = 4 * N_CELLS * sizeof(unsigned int);
    unsigned int * indices = (unsigned int *) malloc(indicesSize);
    for (int y = 0, vidx = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            indices[vidx++] = y * (COLS + 1) + x; // top  left
            indices[vidx++] = y * (COLS + 1) + x + 1; // top right
            indices[vidx++] = (y + 1) * (COLS + 1) + x + 1; // bottom right
            indices[vidx++] = (y + 1) * (COLS + 1) + x; // bottom left
        }
    }

    // for (int y = 0, vidx = 0; y < ROWS + 1; ++y) {
    //     for (int x = 0; x < COLS + 1; ++x) {
    //         int idx = y * (COLS + 1) + x;

    //         printf("%i (%.2f, %.2f)\t", vidx++, vertices[idx].x, vertices[idx].y);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // for (int vidx = 0; vidx < 4 * N_CELLS; ++vidx) {
    //     printf("%u\t", indices[vidx]);
    //     if ((vidx + 1) % 4 == 0) printf("\n");
    // }

    // configure a Vertex Array Object so we configure our objects only once
    glGenVertexArrays(1, &VAO);
    // bind it
    glBindVertexArray(VAO);

    // generate Vertex Buffer Object and store it's ID
    unsigned int VBO;
    glGenBuffers(1, &VBO); 
    // only 1 buffer of a type can be bound simultaneously, that's how OpenGL knows what object we're talking about
    // on each command that refers to the type (GL_ARRAY_BUFFER in this case)
    // bind buffer to context 
    glBindBuffer(GL_ARRAY_BUFFER, VBO);  
    // copy vertext data to buffer
    //  GL_STREAM_DRAW: the data is set only once and used by the GPU at most a few times.
    //  GL_STATIC_DRAW: the data is set only once and used many times.
    //  GL_DYNAMIC_DRAW: the data is changed a lot and used many times.
    glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices, GL_STATIC_DRAW);

    // generate an Element Buffer Object to iterate on the vertices of the VBO
    unsigned int EBO;
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indicesSize, indices, GL_STATIC_DRAW);
    
    // tell OpenGL how to interpret the vertex buffer data
    // params: 
    //      *location* of the position vertex (as in the vertex shader)
    //      size of the vertex attribute, which is a vec3 (size 3)
    //      type of each attribute (vec3 is made of floats)
    //      use_normalization?
    //      stride of each position vertex in the array. It could be 0 as data is tightly packed.
    //      offset in bytes where the data start in the buffer
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    // enable the vertex attributes ixn location 0 for the currently bound VBO
    glEnableVertexAttribArray(0); 
    // unbind VAO to avoid modifications when configuring a new VAO
    glBindVertexArray(0);

    // set mode to wireframe on the front and the back of the polygons
    // this way we can see what are our polygons made of
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    ///// RUN IT

    glutMainLoop();

    ///// CLEAN UP
    // delete VBO when program terminated
    glDeleteBuffers(1, &VBO);

    return 0;
}