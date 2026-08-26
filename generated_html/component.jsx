import React from 'react';

const HelloWorldButton = () => {
  const handleClick = () => {
    alert('Hello, World!');
  };

  return (
    <button onClick={handleClick}>
      Hi
    </button>
  );
};

export default HelloWorldButton;
