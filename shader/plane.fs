#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

// texture sampler
uniform sampler2D texture1;

void main()
{           
  vec3 color = texture(texture1, TexCoord).rgb;
  FragColor = vec4(color, 1.0);
  /* FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f); */
}
