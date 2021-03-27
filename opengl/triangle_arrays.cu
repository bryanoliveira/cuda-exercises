// from https://learnopengl.com/Getting-started/Hello-Triangle
#include <stdio.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#define WIDTH 640
#define HEIGHT 480

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

void display() {
    // use configured shaders
    glUseProgram(shaderProgram);
    // bind VAO, which implicitly binds our VBO
    glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
    // params: 
    //      the OpenGL primitive we will draw
    //      the starting index of the vertex array
    //      how many vertices to draw (a triangle has 3 vertices)
    glDrawArrays(GL_TRIANGLES, 0, 3);
    // unbind VAO to avoid modifications
    // glBindVertexArray(0); // no need to unbind it every time 

    // actually in the screen
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
}

int main(int argc, char **argv) {
    // variables to store shader compiling errors
    int  success;
    char infoLog[512];

    // define triangle vertices
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    }; 


    glutInit(&argc, argv);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("VBO Test");
    glutDisplayFunc(display);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    // this is required to setup GL functions (like glGenBuffers)
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        /* glewInit failed, something is seriously wrong */
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        return 1;
    }


    ///// VBO CONFIGURATION

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
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // unbind buffer so we could configure other VBOs, for example
    glBindBuffer(GL_ARRAY_BUFFER, 0);

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
        return 1;
    }

    ///// SHADER CONFIGURATION

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
        return 1;
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
        return 1;
    }
    // clear already linked shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);


    ///// VAO CONFIGURATION

    // configure a Vertex Array Object so we configure our object only once
    glGenVertexArrays(1, &VAO);
    // bind it
    glBindVertexArray(VAO);

    // configure attributes for rendering
    // the VBO should be bound at this point:
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // tell OpenGL how to interpret the vertex buffer data
    // params: 
    //      *location* of the position vertex (as in the vertex shader)
    //      size of the vertex attribute, which is a vec3 (size 3)
    //      type of each attribute (vec3 is made of floats)
    //      use_normalization?
    //      stride of each position vertex in the array. It could be 0 as data is tightly packed.
    //      offset in bytes where the data start in the buffer
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    // enable the vertex attributes ixn location 0 for the currently bound VBO
    glEnableVertexAttribArray(0); 
    // unbind VAO to avoid modifications when configuring a new VAO
    glBindVertexArray(0);
    // we don't really need to unbind the VBO *after* we unbount the VAO, since it will not store bindings in this case
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // set mode to wireframe on the front and the back of the polygons
    // this way we can see what are our polygons made of
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    ///// RUN IT

    glutMainLoop();

    ///// CLEAN UP
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    return 0;
}