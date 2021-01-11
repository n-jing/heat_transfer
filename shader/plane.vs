#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec2 TexCoord;
out float v_depth;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{

  gl_Position = view * model * vec4(aPos, 1.0);
  v_depth = gl_Position.z;
  gl_Position = projection * gl_Position;
	TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}
