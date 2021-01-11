#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
in float v_depth;

// texture sampler
uniform sampler2D texture1;

void main()
{           
  vec3 origin_color = texture(texture1, TexCoord).rgb;


  float blur = 0;
  float dis = 20.0;
  float near = -18.0;
  float far = -12.0;

  if(v_depth >= near && v_depth <= far)
    blur = 0;
  else if(v_depth < near)
  {
    float no = abs(v_depth - near) / dis;
    blur = clamp(no, 0.0f, 1.0f);
  }
  else
    blur = 0.0f;

  vec3 color = mix(vec3(origin_color), vec3(1.0f, 1.0f, 1.0f), blur);

  FragColor = vec4(color, 1.0);
  /* FragColor = vec4(1.0f, 1.0f, 1.0f, 1.0f); */
}
