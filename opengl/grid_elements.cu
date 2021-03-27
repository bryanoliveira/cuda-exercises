// from https://learnopengl.com/Getting-started/Hello-Triangle
#include <stdio.h>
#include <time.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#define WIDTH 640
#define HEIGHT 480
#define ROWS 50
#define COLS 50

#define N_CELLS ROWS * COLS
// we define disjunct vertexes for each cell so we can color them independently
#define N_VERTEX 4 * ROWS * COLS


typedef struct _vec3 {
    float x, y, z;
    float state;
    _vec3(float _x, float _y, float _z, float _s) : x(_x), y(_y), z(_z), state(_s) {};
} vec3;

const char *vertexShaderSource = "#version 460 core\n"
                                 "layout (location = 0) in vec3 aPos;\n"
                                 "layout (location = 1) in float state;\n"
                                 "out float v_state;\n"
                                 "void main() {\n"
                                 "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
                                 "   v_state = state;\n"
                                 "}\0";

const char *fragmentShaderSource = "#version 460 core\n"
                                   "in float v_state;\n"
                                   "out vec4 FragColor;\n"
                                   "void main() {\n"
                                   "    FragColor = vec4(v_state, v_state, v_state, v_state);\n"
                                   "}\0";

unsigned int shaderProgram;
unsigned int VAO;
vec3 * vertices;
size_t verticesSize;


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
    glDrawElements(GL_QUADS, N_VERTEX, GL_UNSIGNED_INT, 0);
    // glBindVertexArray(0); // no need to unbind it every time 

    float state = float(rand()) / RAND_MAX > 0.5 ? 1.0f : 0.0f;
    for (int vidx = 0; vidx < N_VERTEX; ++vidx) {
        vertices[vidx].state = state;
        if ((vidx + 1) % 4 == 0) 
            state = float(rand()) / RAND_MAX > 0.5 ? 1.0f : 0.0f;
    }
    glBufferData(GL_ARRAY_BUFFER, verticesSize, vertices, GL_STATIC_DRAW);

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
    srand(time(NULL));

    ///// INIT GL
    initGL(&argc, argv);

    ///// SHADER CONFIGURATION
    setupShaderProgram();
    

    ///// OBJECT & BUFFER CONFIGURATION

    // define square vertices
    verticesSize = N_VERTEX * sizeof(vec3);
    vertices = (vec3 *) malloc(verticesSize);
    // square indexes
    size_t indicesSize = N_VERTEX * sizeof(unsigned int);
    unsigned int * indices = (unsigned int *) malloc(indicesSize);
    // iterate over the number of cells
    for (int y = 0, idx = 0; y < ROWS; ++y) {
        for (int x = 0; x < COLS; ++x) {
            // int idx = y * COLS + x;
            float state = float(rand()) / RAND_MAX > 0.5 ? 1.0f : 0.0f;

            // vertices live in an (-1, 1) tridimensional space
            // we need to calculate the position of each vertice inside a 2d grid
            // top left
            vertices[idx] = vec3(-1.0f + x * (2.0 / COLS), -1.0f + y * (2.0 / ROWS), 0.0f, state);
            indices[idx] = idx;
            idx++;
            // top right
            vertices[idx] = vec3(-1.0f + (x + 1) * (2.0 / COLS), -1.0f + y * (2.0 / ROWS), 0.0f, state);
            indices[idx] = idx;
            idx++;
            // bottom right
            vertices[idx] = vec3(-1.0f + (x + 1) * (2.0 / COLS), -1.0f + (y + 1) * (2.0 / ROWS), 0.0f, state);
            indices[idx] = idx;
            idx++;
            // bottom left
            vertices[idx] = vec3(-1.0f + x * (2.0 / COLS), -1.0f + (y + 1) * (2.0 / ROWS), 0.0f, state);
            indices[idx] = idx;
            idx++;
        }
    }

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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(vec3), (void*)(3*sizeof(float)));
    // enable the vertex attributes ixn location 0 for the currently bound VBO
    glEnableVertexAttribArray(0); 
    glEnableVertexAttribArray(1);
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